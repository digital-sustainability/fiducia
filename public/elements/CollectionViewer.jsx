import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AlertTriangle, Folder, Search, Trash2 } from 'lucide-react';
import { useState } from 'react';

// Props are globally injected by Chainlit.
// The `props` variable is available directly in the component's scope.

export default function CollectionViewer() {
    const { collections = {}, collectionStats = {} } = props || {};
    const [deleteCollectionDialog, setDeleteCollectionDialog] = useState({ open: false, collectionName: '' });
    const [deleteFileDialog, setDeleteFileDialog] = useState({ open: false, collectionName: '', filename: '' });
    const [searchDialog, setSearchDialog] = useState({ open: false, collectionName: '', query: '' });

    // Function to handle collection search
    const handleSearchCollection = async (collectionName, query) => {
        try {
            const action = {
                name: "search_collection",
                payload: {
                    collection_name: collectionName,
                    query: query,
                }
            };
            await callAction(action);
            setSearchDialog({ open: false, collectionName: '', query: '' });
        } catch (error) {
            console.error('Failed to search collection:', error);
        }
    };

    // Function to handle collection deletion confirmation
    const handleDeleteCollection = async (collectionName) => {
        try {
            const action = {
                name: "delete_collection",
                payload: {
                    collection_name: collectionName,
                }
            };
            await callAction(action);
            setDeleteCollectionDialog({ open: false, collectionName: '' });
        } catch (error) {
            console.error('Failed to delete collection:', error);
        }
    };

    // Function to handle individual file deletion confirmation
    const handleDeleteFile = async (collectionName, filename) => {
        try {
            const action = {
                name: "delete_file_from_collection",
                payload: {
                    collection_name: collectionName,
                    filename: filename,
                }
            };
            await callAction(action);
            setDeleteFileDialog({ open: false, collectionName: '', filename: '' });
        } catch (error) {
            console.error('Failed to delete file:', error);
        }
    };

    // Get file extension icon
    const getFileIcon = (filename) => {
        const extension = filename.split('.').pop()?.toLowerCase();
        switch (extension) {
            case 'pdf':
                return '📄';
            case 'doc':
            case 'docx':
                return '📝';
            case 'txt':
                return '📄';
            default:
                return '📄';
        }
    };

    const collectionNames = Object.keys(collectionStats);

    if (!collectionNames || collectionNames.length === 0) {
        return (
            <Card className="w-full max-w-4xl mx-auto p-4 shadow-lg bg-card text-card-foreground">
                <CardHeader>
                    <CardTitle className="text-xl font-bold flex items-center gap-2">
                        <Folder className="h-6 w-6" />
                        Document Collections
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-muted-foreground">No document collections found. Use the 'Add Collection' command to add some documents!</p>
                </CardContent>
            </Card>
        );
    }

    return (
        <div className="w-full max-w-6xl mx-auto p-4 space-y-4">
            <Card className="shadow-lg bg-card text-card-foreground">
                <CardHeader>
                    <CardTitle className="text-xl font-bold flex items-center gap-2">
                        <Folder className="h-6 w-6" />
                        Document Collections ({collectionNames.length})
                    </CardTitle>
                </CardHeader>
            </Card>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {collectionNames.map((collectionName) => {
                    const stats = collectionStats[collectionName] || {};
                    const fileCount = stats.file_count || 0;
                    const docCount = stats.document_count || 0;
                    const fileTypes = stats.file_types || [];
                    const files = stats.files || [];

                    return (
                        <Card key={collectionName} className="shadow-lg bg-card text-card-foreground hover:shadow-xl transition-shadow">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg font-semibold flex items-center justify-between">
                                    <span className="flex items-center gap-2">
                                        <Folder className="h-5 w-5" />
                                        {collectionName}
                                    </span>
                                </CardTitle>
                                <div className="text-sm text-muted-foreground">
                                    <p>{fileCount} files • {docCount} chunks</p>
                                    {fileTypes.length > 0 && (
                                        <p>Types: {fileTypes.join(', ')}</p>
                                    )}
                                </div>
                            </CardHeader>

                            <CardContent className="pt-0">
                                {/* Files list */}
                                <div className="space-y-2 mb-4 max-h-32 overflow-y-auto">
                                    {files.slice(0, 5).map((filename, index) => (
                                        <div key={index} className="flex items-center justify-between text-sm p-2 rounded bg-muted/30">
                                            <span className="flex items-center gap-2 flex-1 min-w-0">
                                                <span className="text-lg">{getFileIcon(filename)}</span>
                                                <span className="truncate" title={filename}>{filename}</span>
                                            </span>
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                                                onClick={() => setDeleteFileDialog({
                                                    open: true,
                                                    collectionName: collectionName,
                                                    filename: filename
                                                })}
                                                title={`Delete ${filename}`}
                                            >
                                                <Trash2 className="h-3 w-3" />
                                            </Button>
                                        </div>
                                    ))}
                                    {files.length > 5 && (
                                        <p className="text-xs text-muted-foreground text-center">
                                            ...and {files.length - 5} more files
                                        </p>
                                    )}
                                </div>

                                {/* Action buttons */}
                                <div className="flex gap-2">
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        className="flex-1"
                                        onClick={() => setSearchDialog({
                                            open: true,
                                            collectionName: collectionName,
                                            query: ''
                                        })}
                                    >
                                        <Search className="h-4 w-4 mr-1" />
                                        Search
                                    </Button>

                                    <Button
                                        variant="outline"
                                        size="sm"
                                        className="text-destructive hover:text-destructive"
                                        onClick={() => setDeleteCollectionDialog({
                                            open: true,
                                            collectionName: collectionName
                                        })}
                                    >
                                        <Trash2 className="h-4 w-4" />
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    );
                })}
            </div>

            {/* Collection Delete Confirmation Dialog */}
            <Dialog open={deleteCollectionDialog.open} onOpenChange={(open) =>
                setDeleteCollectionDialog({ open, collectionName: '' })
            }>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            <AlertTriangle className="h-5 w-5 text-destructive" />
                            Delete Collection
                        </DialogTitle>
                        <DialogDescription>
                            Are you sure you want to delete the collection <strong>"{deleteCollectionDialog.collectionName}"</strong>?
                            <br /><br />
                            <strong>This action cannot be undone and will remove:</strong>
                            <ul className="list-disc list-inside mt-2 space-y-1">
                                <li>All documents in the collection</li>
                                <li>All associated metadata</li>
                                <li>All search indices for this collection</li>
                            </ul>
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button
                            variant="outline"
                            onClick={() => setDeleteCollectionDialog({ open: false, collectionName: '' })}
                        >
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={() => handleDeleteCollection(deleteCollectionDialog.collectionName)}
                        >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete Collection
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* File Delete Confirmation Dialog */}
            <Dialog open={deleteFileDialog.open} onOpenChange={(open) =>
                setDeleteFileDialog({ open, collectionName: '', filename: '' })
            }>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            <AlertTriangle className="h-5 w-5 text-destructive" />
                            Delete File
                        </DialogTitle>
                        <DialogDescription>
                            Are you sure you want to delete the file <strong>"{deleteFileDialog.filename}"</strong> from collection <strong>"{deleteFileDialog.collectionName}"</strong>?
                            <br /><br />
                            <strong>This action will remove:</strong>
                            <ul className="list-disc list-inside mt-2 space-y-1">
                                <li>All document chunks from this file</li>
                                <li>All associated metadata</li>
                                <li>All search indices for this file</li>
                            </ul>
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button
                            variant="outline"
                            onClick={() => setDeleteFileDialog({ open: false, collectionName: '', filename: '' })}
                        >
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={() => handleDeleteFile(deleteFileDialog.collectionName, deleteFileDialog.filename)}
                        >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete File
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Search Dialog */}
            <Dialog open={searchDialog.open} onOpenChange={(open) =>
                setSearchDialog({ open, collectionName: '', query: '' })
            }>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            <Search className="h-5 w-5 text-primary" />
                            Search Collection
                        </DialogTitle>
                        <DialogDescription>
                            Enter your search query to find relevant documents in the collection <strong>"{searchDialog.collectionName}"</strong>.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-4 py-4">
                        <div className="grid gap-2">
                            <Label htmlFor="search-query">Search Query</Label>
                            <Input
                                id="search-query"
                                placeholder="Enter your search terms..."
                                value={searchDialog.query}
                                onChange={(e) => setSearchDialog(prev => ({ ...prev, query: e.target.value }))}
                                onKeyPress={(e) => {
                                    if (e.key === 'Enter' && searchDialog.query.trim()) {
                                        handleSearchCollection(searchDialog.collectionName, searchDialog.query);
                                    }
                                }}
                            />
                        </div>
                    </div>
                    <DialogFooter>
                        <Button
                            variant="outline"
                            onClick={() => setSearchDialog({ open: false, collectionName: '', query: '' })}
                        >
                            Cancel
                        </Button>
                        <Button
                            onClick={() => handleSearchCollection(searchDialog.collectionName, searchDialog.query)}
                            disabled={!searchDialog.query.trim()}
                        >
                            <Search className="h-4 w-4 mr-2" />
                            Search
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
