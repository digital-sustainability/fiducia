import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Calendar, ExternalLink, FileText, Tag, Users } from 'lucide-react';

// Props are globally injected by Chainlit.
// The `props` variable is available directly in the component's scope.

export default function TimelineViewer() {
    const { timelineEvents = [], collectionName = 'N/A' } = props || {};

    // Create a mapping from event types to hex color codes for inline styling.
    const eventTypeColorMap = {
        judgment: '#830074ff',
        filing: '#155bcbff',
        conciliation_request: '#155bcbff',
        indictment: '#155bcbff',
        hearing: '#eab308', // yellow-500
        order: '#8b5cf6', // purple-500
        deadline: '#8b5cf6', // purple-500
        penalty_order: '#8b5cf6', // purple-500
        appeal: '#f97316', // orange-500
        settlement: '#22c55e', // green-500
        other: '#9ca3af' // gray-400
    };

    // Function to handle opening a PDF document
    const handleOpenDocument = async (event) => {
        if (!event.source_path || !event.source_file) {
            console.warn('No source information available for this event');
            return;
        }

        try {
            const action = {
                name: "open_pdf_document",
                payload: {
                    file_path: event.source_path,
                    filename: event.source_file,
                    page: event.page_start || 1, // Use page_start from event or default to 1
                }
            };

            // Call the Python function using Chainlit's correct API method
            await callAction(action);
        } catch (error) {
            console.error('Failed to open document:', error);
        }
    };

    if (!timelineEvents || timelineEvents.length === 0) {
        return (
            <Card className="w-full max-w-2xl mx-auto p-4 shadow-lg bg-card text-card-foreground">
                <CardHeader>
                    <CardTitle className="text-xl font-bold">Timeline for "{collectionName}"</CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-muted-foreground">No timeline events could be found for this collection.</p>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="w-full max-w-2xl mx-auto p-4 shadow-lg bg-card text-card-foreground">
            <CardHeader>
                <CardTitle className="text-xl font-bold">Timeline for "{collectionName}"</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    {timelineEvents.map((event, index) => {
                        const eventType = event.event_type?.toLowerCase() || 'other';
                        const color = eventTypeColorMap[eventType] || eventTypeColorMap.other;

                        return (
                            <div key={index} className="p-4 rounded-lg border bg-card shadow">
                                <div className="flex items-center gap-2">
                                    <div
                                        className="h-3 w-3 mr-3 pr-3 rounded-full flex-shrink-0"
                                        style={{ backgroundColor: color }}
                                    ></div>
                                    <div className="flex items-center gap-2">
                                        <Calendar className="h-4 w-4 text-muted-foreground" />
                                        <span className="font-semibold">{event.date}</span>
                                    </div>
                                </div>
                                <div className="pl-7 mt-2">
                                    <div className="flex items-start gap-2 mb-2">
                                        <FileText className="h-5 w-5 mt-0.5 text-muted-foreground flex-shrink-0" />
                                        <p className="font-medium text-card-foreground">{event.label}</p>
                                    </div>
                                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                        <Tag className="h-4 w-4 flex-shrink-0" />
                                        <span>Type: <span className="font-semibold">{event.event_type}</span></span>
                                    </div>
                                    <div className="flex items-center gap-2 text-sm text-muted-foreground mt-1">
                                        <Users className="h-4 w-4 flex-shrink-0" />
                                        <span>Participants: <span className="font-semibold">{event.participants.join(', ') || 'N/A'}</span></span>
                                    </div>
                                    {event.source_file && (
                                        <div className="flex items-center gap-2 text-sm text-muted-foreground mt-2">
                                            <FileText className="h-4 w-4 flex-shrink-0" />
                                            <span>Source: </span>
                                            <button
                                                onClick={() => handleOpenDocument(event)}
                                                className="font-semibold text-blue-600 hover:text-blue-800 hover:underline flex items-center gap-1 transition-colors"
                                                title={`Open ${event.source_file}`}
                                            >
                                                {event.source_file} {event.page_start !== event.page_end ? `(Pages ${event.page_start}-${event.page_end})` : `(Page ${event.page_start})`}
                                                <ExternalLink className="h-3 w-3" />
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </CardContent>
        </Card>
    );
}
