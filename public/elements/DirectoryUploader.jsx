import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Check, X } from 'lucide-react';
import { useState } from 'react';

export default function DirectoryUploader() {
    const { userLanguage = 'de', translations = {}, userMessage = '' } = props || {};

    const [selectedFiles, setSelectedFiles] = useState([]);
    const [collectionName, setCollectionName] = useState(userMessage);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState(null);
    const [errorMessage, setErrorMessage] = useState('');
    const [embeddingStatus, setEmbeddingStatus] = useState(null);
    const [embeddingProgress, setEmbeddingProgress] = useState('');

    const t = (key, options = {}) => {
        let translatedText = translations[key] || key;
        for (const optionKey in options) {
            if (options.hasOwnProperty(optionKey)) {
                const regex = new RegExp(`{{\\s*${optionKey}\\s*}}`, 'g');
                translatedText = translatedText.replace(regex, options[optionKey]);
            }
        }
        return translatedText;
    };

    const allowedExtensions = ['.pdf', '.doc', '.docx', '.txt', '.md', '.csv', '.json', '.xml'];

    const handleFileChange = (event) => {
        const files = Array.from(event.target.files);
        const validFiles = files.filter(file => {
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            return allowedExtensions.includes(ext);
        });
        setSelectedFiles(validFiles);
        setUploadProgress(0);
        setUploadStatus(null);
        setErrorMessage('');
    };

    const handleUpload = async () => {
        if (selectedFiles.length === 0) {
            setErrorMessage(t("directory_uploader_error_no_directory"));
            return;
        }
        if (!collectionName.trim()) {
            setErrorMessage(t("directory_uploader_error_no_collection_name"));
            return;
        }

        setIsUploading(true);
        setUploadProgress(0);
        setUploadStatus(null);
        setErrorMessage('');
        setEmbeddingStatus(null);
        setEmbeddingProgress('');

        try {
            const filesData = await Promise.all(
                selectedFiles.map(async (file) => {
                    return new Promise((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onload = () => {
                            const base64 = reader.result.split(',')[1];
                            resolve({
                                name: file.name,
                                relativePath: file.webkitRelativePath || file.name,
                                content: base64,
                                size: file.size
                            });
                        };
                        reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
                        reader.readAsDataURL(file);
                    });
                })
            );

            const result = await callAction({
                name: "upload_documents",
                payload: {
                    collectionName: collectionName.trim(),
                    files: filesData
                }
            });

            if (result.success) {
                console.log("[DEBUG] Upload successful, result:", result);
                setUploadProgress(100);
                setUploadStatus('success');

                callAction({
                    name: "send_toast",
                    payload: {
                        "message": t("directory_uploader_success_toast_message", {
                            count: selectedFiles.length,
                            collectionName: collectionName.trim()
                        }),
                        "type": "success"
                    }
                });

                // Check for embedding_queued flag in multiple possible locations
                const embeddingQueued = result.response?.embedding_queued ||
                    result.embedding_queued ||
                    result.response?.response?.embedding_queued;

                console.log("[DEBUG] Checking embedding_queued:", {
                    "result.response?.embedding_queued": result.response?.embedding_queued,
                    "result.embedding_queued": result.embedding_queued,
                    "result.response?.response?.embedding_queued": result.response?.response?.embedding_queued,
                    "final embeddingQueued": embeddingQueued
                });

                if (embeddingQueued) {
                    console.log("[DEBUG] Setting embedding status to queued and starting polling");
                    setEmbeddingStatus('queued');

                    // Set helpful initial message based on collection size
                    let initialMessage = 'Starting document processing...';
                    if (selectedFiles.length > 20) {
                        initialMessage = `Processing ${selectedFiles.length} files - this will take 1-3 hours. You can safely close this page.`;
                    } else if (selectedFiles.length > 10) {
                        initialMessage = `Processing ${selectedFiles.length} files - this will take 30-60 minutes. You can safely close this page.`;
                    } else if (selectedFiles.length > 5) {
                        initialMessage = `Processing ${selectedFiles.length} files - this will take 10-30 minutes. You can safely close this page.`;
                    } else {
                        initialMessage = `Processing ${selectedFiles.length} files - this will take 5-15 minutes.`;
                    }

                    setEmbeddingProgress(initialMessage);
                    pollEmbeddingStatus(collectionName.trim());
                } else {
                    console.log("[DEBUG] No embedding_queued flag found, not starting polling");
                }
            } else {
                setUploadStatus('error');
                setErrorMessage(result.error || "Upload failed");
            }
        } catch (error) {
            setUploadStatus('error');
            setErrorMessage(`${t("directory_uploader_network_error_prefix")} ${error.message}`);
        } finally {
            setIsUploading(false);
            callAction({ name: "refresh_available_collections", payload: {} });
        }
    };

    const pollEmbeddingStatus = async (collectionName) => {
        console.log(`[DEBUG] Starting to poll embedding status for collection: ${collectionName}`);

        // Calculate dynamic timeout based on file count and sizes
        const totalSizeMB = selectedFiles.reduce((sum, file) => sum + (file.size / 1024 / 1024), 0);
        const avgSizeMB = totalSizeMB / selectedFiles.length;

        // More refined time estimates based on actual performance data
        let estimatedMinutesPerFile;
        if (avgSizeMB < 0.5) {
            estimatedMinutesPerFile = 1.0; // Very small files: 1 min each
        } else if (avgSizeMB < 1) {
            estimatedMinutesPerFile = 1.5; // Small files (< 1MB): 1.5 min each
        } else if (avgSizeMB < 3) {
            estimatedMinutesPerFile = 2.5; // Medium files (1-3MB): 2.5 min each
        } else if (avgSizeMB < 8) {
            estimatedMinutesPerFile = 5.0; // Large files (3-8MB): 5 min each
        } else if (avgSizeMB < 20) {
            estimatedMinutesPerFile = 8.0; // Very large files (8-20MB): 8 min each
        } else {
            estimatedMinutesPerFile = 12.0; // Huge files (>20MB): 12 min each
        }

        // Add complexity overhead for large collections
        let complexityMultiplier = 1.0;
        if (selectedFiles.length > 20) {
            complexityMultiplier = 1.3; // 30% overhead for large collections
        } else if (selectedFiles.length > 50) {
            complexityMultiplier = 1.5; // 50% overhead for very large collections
        }

        // Apply buffer for document complexity and system variations
        const bufferMultiplier = 1.8; // 80% buffer for variability
        const baseTimeout = Math.ceil(selectedFiles.length * estimatedMinutesPerFile * bufferMultiplier * complexityMultiplier);

        const minimumTimeoutMinutes = 15; // At least 15 minutes
        const maxTimeoutMinutes = 600; // Cap at 10 hours for very large collections

        let timeoutMinutes = Math.max(minimumTimeoutMinutes, Math.min(baseTimeout, maxTimeoutMinutes));

        const maxAttempts = Math.floor((timeoutMinutes * 60) / 2); // 2 second intervals initially
        console.log(`[DEBUG] File analysis: ${selectedFiles.length} files, avg ${avgSizeMB.toFixed(1)}MB, total ${totalSizeMB.toFixed(1)}MB`);
        console.log(`[DEBUG] Calculated timeout: ${timeoutMinutes} minutes (${maxAttempts} attempts), estimated ${estimatedMinutesPerFile}min/file (complexity: ${complexityMultiplier}x)`);
        console.log(`[DEBUG] Expected completion in ~${Math.round(selectedFiles.length * estimatedMinutesPerFile * complexityMultiplier)} minutes`);

        let attempts = 0;
        let consecutiveErrors = 0;
        const maxConsecutiveErrors = 8; // Increased for very long processes
        let lastStatusTime = Date.now(); const poll = async () => {
            const progressPercent = Math.min(Math.floor((attempts / maxAttempts) * 100), 99);
            console.log(`[DEBUG] Poll attempt ${attempts + 1}/${maxAttempts} (${progressPercent}% of max time)`);

            try {
                const timeoutPromise = new Promise((_, reject) =>
                    setTimeout(() => reject(new Error('Request timeout')), 8000) // Longer request timeout
                );

                const actionPromise = callAction({
                    name: "get_embedding_status",
                    payload: { collectionName }
                });

                const result = await Promise.race([actionPromise, timeoutPromise]);
                console.log(`[DEBUG] Poll result:`, result);

                let status = null;
                if (result?.success && result?.response) {
                    status = result.response;
                    console.log(`[DEBUG] Using result.response:`, status);
                    console.log(`[DEBUG] Status.status value:`, status.status, `(type: ${typeof status.status})`);
                } else if (result?.status) {
                    status = result;
                    console.log(`[DEBUG] Using result directly:`, status);
                    console.log(`[DEBUG] Status.status value:`, status.status, `(type: ${typeof status.status})`);
                } else {
                    console.log(`[DEBUG] Invalid response format:`, result);
                    throw new Error('Invalid response format');
                }

                consecutiveErrors = 0;
                console.log(`[DEBUG] Processing status:`, status.status, `(type: ${typeof status.status})`);
                console.log(`[DEBUG] Full status object:`, JSON.stringify(status, null, 2));

                // Extract the actual status value more defensively
                const actualStatus = status?.status || status?.response?.status || 'unknown';
                console.log(`[DEBUG] Extracted actual status:`, actualStatus, `(type: ${typeof actualStatus})`);

                // Handle case where status is not available
                if (!actualStatus || actualStatus === 'unknown') {
                    console.log(`[DEBUG] ⚠️ No valid status found. Full object:`, status);
                    setEmbeddingProgress('Status unavailable, retrying...');
                    // Continue polling
                    attempts++;
                    if (attempts < maxAttempts) {
                        const interval = attempts < 150 ? 2000 : attempts < 300 ? 5000 : 10000;
                        setTimeout(poll, interval);
                    } else {
                        setEmbeddingStatus('error');
                        setEmbeddingProgress(`⏰ Unable to get status after ${timeoutMinutes} minutes. Processing may still be running - check collections manually.`);
                    }
                    return;
                }

                // Normalize status string (trim whitespace, convert to lowercase)
                const normalizedStatus = actualStatus.toString().trim().toLowerCase();
                console.log(`[DEBUG] Normalized status:`, normalizedStatus);

                // Check for completed status with multiple fallback comparisons
                if (normalizedStatus === 'completed' || actualStatus === 'completed') {
                    console.log(`[DEBUG] ✅ Embedding completed! Setting UI to completed state (matched: ${normalizedStatus})`);
                    setEmbeddingStatus('completed');
                    const finalMessage = status.elapsed_display ?
                        `✅ Completed in ${status.elapsed_display}! Documents are now searchable.` :
                        `✅ Processing completed! Documents are now searchable.`;
                    setEmbeddingProgress(finalMessage);

                    callAction({
                        name: "send_toast",
                        payload: {
                            "message": `Processing completed for collection "${collectionName}". Documents are now searchable!`,
                            "type": "success"
                        }
                    });
                    callAction({ name: "reload_vector_store", payload: {} });
                    return;
                } else if (normalizedStatus === 'error' || actualStatus === 'error') {
                    console.log(`[DEBUG] ❌ Processing error detected`);
                    setEmbeddingStatus('error');
                    setEmbeddingProgress(`❌ Processing failed: ${status.message}`);
                    return;
                } else if (normalizedStatus === 'processing' || actualStatus === 'processing') {
                    console.log(`[DEBUG] 🔄 Still processing...`);
                    setEmbeddingStatus('processing');
                    let progressMessage = status.message || 'Processing documents...';

                    // Add more informative progress details
                    if (status.elapsed_display) {
                        progressMessage += ` (${status.elapsed_display} elapsed)`;
                    }
                    if (status.total_files) {
                        progressMessage += ` - ${status.total_files} files`;

                        // Enhanced time estimation based on actual processing speed
                        if (status.elapsed_time && status.elapsed_time > 300) { // After 5 minutes, start estimating
                            // Try to determine how many files have been processed
                            // This is a rough estimate based on polling frequency
                            const pollingMinutes = status.elapsed_time / 60;
                            const estimatedFilesProcessed = Math.max(1, Math.floor(pollingMinutes / estimatedMinutesPerFile));

                            if (estimatedFilesProcessed < status.total_files) {
                                const actualTimePerFile = status.elapsed_time / estimatedFilesProcessed;
                                const remainingFiles = status.total_files - estimatedFilesProcessed;
                                const estimatedRemainingTime = remainingFiles * actualTimePerFile;

                                if (estimatedRemainingTime > 3600) { // More than 1 hour
                                    const hours = Math.floor(estimatedRemainingTime / 3600);
                                    const minutes = Math.floor((estimatedRemainingTime % 3600) / 60);
                                    progressMessage += ` (~${hours}h ${minutes}m remaining)`;
                                } else if (estimatedRemainingTime > 60) {
                                    const minutes = Math.floor(estimatedRemainingTime / 60);
                                    progressMessage += ` (~${minutes}m remaining)`;
                                }

                                // Show processing rate
                                const minutesPerFile = Math.round(actualTimePerFile / 60 * 10) / 10;
                                progressMessage += ` [${minutesPerFile}min/file]`;
                            }
                        }
                    }

                    // Adaptive timeout warning based on actual progress
                    const timeoutPercent = Math.floor((attempts / maxAttempts) * 100);
                    if (timeoutPercent > 85) {
                        progressMessage += ` ⚠️ Approaching timeout - processing continues in background`;
                    } else if (timeoutPercent > 70) {
                        progressMessage += ` ⚠️ Long processing expected for large documents`;
                    }

                    setEmbeddingProgress(progressMessage);
                    lastStatusTime = Date.now(); // Update last activity time
                } else {
                    console.log(`[DEBUG] ⚠️ Unknown status "${normalizedStatus}" (original: "${actualStatus}", type: ${typeof actualStatus}). Full status:`, status);
                    setEmbeddingStatus('queued');
                    setEmbeddingProgress(`Status: ${actualStatus || 'unknown'} - Processing in progress...`);
                }
            } catch (error) {
                consecutiveErrors++;

                if (consecutiveErrors >= maxConsecutiveErrors) {
                    setEmbeddingStatus('error');
                    setEmbeddingProgress(`❌ Connection lost after ${consecutiveErrors} errors. Processing may still be running - check collections manually.`);
                    return;
                }

                setEmbeddingProgress(`⚠️ Connection issue (${consecutiveErrors}/${maxConsecutiveErrors}). Retrying...`);
            }

            attempts++;
            if (attempts < maxAttempts) {
                // Use longer intervals as time goes on to reduce server load
                const interval = attempts < 150 ? 2000 : attempts < 300 ? 5000 : 10000;
                setTimeout(poll, interval);
            } else {
                setEmbeddingStatus('error');
                const timeoutMessage = selectedFiles.length > 10
                    ? `⏰ Large collection (${selectedFiles.length} files) is taking longer than expected (${timeoutMinutes} minutes). Processing continues in background - check "Show Collections" periodically to see when it's completed.`
                    : `⏰ Processing timed out after ${timeoutMinutes} minutes. Check "Show Collections" to see if processing completed.`;
                setEmbeddingProgress(timeoutMessage);
            }
        };

        poll();
    };

    return (
        <Card className="w-full p-4 shadow-lg bg-card text-card-foreground">
            <CardHeader className="pb-4">
                <CardTitle className="text-xl font-bold">{t("directory_uploader_card_title")}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                <div>
                    <label htmlFor="collection-name-input" className="block text-sm font-medium text-muted-foreground mb-2">
                        {t("directory_uploader_collection_name_label")}
                    </label>
                    <Input
                        id="collection-name-input"
                        type="text"
                        placeholder={t("directory_uploader_collection_name_placeholder")}
                        value={collectionName}
                        onChange={(e) => setCollectionName(e.target.value)}
                        className="mb-4"
                        disabled={isUploading}
                    />

                    <label htmlFor="directory-input" className="block text-sm font-medium text-muted-foreground mb-2">
                        {t("directory_uploader_select_directory_label")}
                    </label>
                    <Input
                        id="directory-input"
                        type="file"
                        webkitdirectory=""
                        mozdirectory=""
                        onChange={handleFileChange}
                        className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 cursor-pointer"
                        disabled={isUploading}
                    />
                    <p className="text-xs text-muted-foreground mt-2 mb-2">
                        {t("directory_uploader_scan_for_info", { extensions: allowedExtensions.map(ext => ext.toUpperCase()).join(', ') })}
                    </p>
                </div>

                {selectedFiles.length > 0 && (
                    <div className="border rounded-md p-3 bg-accent/20">
                        <h3 className="text-md font-semibold mb-3">
                            {t("directory_uploader_found_documents_header", { count: selectedFiles.length })}
                        </h3>
                        <ScrollArea className="h-40 w-full">
                            <ul className="list-none p-0 m-0 text-sm">
                                {selectedFiles.map((file, index) => (
                                    <li key={index} className="flex justify-between items-center py-1.5 border-b last:border-b-0 border-border">
                                        <span className="truncate pr-2" title={file.webkitRelativePath || file.name}>
                                            {file.webkitRelativePath || file.name}
                                        </span>
                                        <span className="text-xs text-muted-foreground flex-shrink-0">
                                            ({(file.size / 1024 / 1024).toFixed(2)} MB)
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        </ScrollArea>
                    </div>
                )}

                <div className="flex flex-col space-y-4">
                    <Button
                        onClick={handleUpload}
                        className="w-full mt-2 py-2.5 text-base"
                        disabled={uploadStatus === 'success' || isUploading || selectedFiles.length === 0 || !collectionName.trim()}
                    >
                        {isUploading ? t("directory_uploader_uploading_button") : t("directory_uploader_upload_button")}
                    </Button>

                    {isUploading && (
                        <Progress value={uploadProgress} className="w-full mt-2 h-2.5" />
                    )}

                    {uploadStatus === 'success' && (
                        <div className="flex items-center text-green-600 dark:text-green-400 mt-3 text-sm font-medium">
                            <Check className="h-5 w-5 mr-2" /> {t("directory_uploader_upload_success_message")}
                        </div>
                    )}

                    {embeddingStatus && (
                        <div className="mt-3 p-3 border rounded-md bg-accent/10">
                            <div className="text-sm font-medium mb-2">Document Processing Status:</div>
                            <div className={`text-sm mb-2 ${embeddingStatus === 'completed' ? 'text-green-600 dark:text-green-400' :
                                embeddingStatus === 'error' ? 'text-red-600 dark:text-red-400' :
                                    'text-blue-600 dark:text-blue-400'
                                }`}>
                                {embeddingStatus === 'processing' && <span className="animate-pulse">🔄 </span>}
                                {embeddingStatus === 'queued' && <span className="animate-pulse">⏳ </span>}
                                {embeddingStatus === 'completed' && <span>✅ </span>}
                                {embeddingStatus === 'error' && <span>❌ </span>}
                                {embeddingProgress}
                            </div>
                            {(embeddingStatus === 'processing' || embeddingStatus === 'queued') && selectedFiles.length > 5 && (
                                <div className="text-xs text-muted-foreground mt-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded border-l-2 border-blue-300">
                                    💡 <strong>Large collection detected:</strong> Processing includes document parsing, AI metadata enrichment, and vector embedding.
                                    You can safely close this page - processing continues in the background.
                                    Check "Show Collections" to see when it's completed.
                                </div>
                            )}
                        </div>
                    )}

                    {uploadStatus === 'error' && (
                        <div className="flex items-center text-red-600 dark:text-red-400 mt-3 text-sm font-medium">
                            <X className="h-5 w-5 mr-2" /> {errorMessage}
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}