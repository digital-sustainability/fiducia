# Show Facts Feature Documentation

## Overview

The "Show Facts" command allows users to extract and view key facts from document collections using LLM-powered analysis. This feature processes all documents in a collection and generates a comprehensive facts summary that is cached for future use.

## How It Works

### Architecture

1. **CollectionMetadataManager**: A new class that manages fact summaries using SQLite storage
2. **SQLite Database**: Stores generated facts summaries with collection name as primary key
3. **LLM Integration**: Uses an OpenAI-compatible generator configured for Together-hosted Llama models
4. **Chunk Processing**: Iteratively processes document chunks to build comprehensive summaries

### Process Flow

1. **Command Invocation**: User types "Show Facts [collection_name]"
2. **Validation**: System checks if collection exists
3. **Cache Check**: Looks for existing facts in SQLite database
4. **Generation (if needed)**:
   - Retrieves all document chunks from vector store
   - Groups chunks by filename and orders them properly
   - Iteratively extracts facts using LLM calls
   - Creates final comprehensive summary
   - Stores result in database
5. **Display**: Shows facts summary to user with progress updates

## Usage

### Command Syntax
```
Show Facts [collection_name]
```

### Examples
```
Show Facts Empfehlungen
Show Facts Free-Speech
```

### Error Handling
- Collection name validation
- Progress updates during generation
- Graceful error recovery
- Fallback chunk ordering

## Database Schema

### Facts Table
```sql
CREATE TABLE facts (
    collection_name TEXT PRIMARY KEY,
    facts_summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## Features

### Real-time Progress Updates
- File processing status
- Chunk processing progress
- Percentage completion
- Status messages during LLM calls

### Intelligent Fact Extraction
- Parties and stakeholders identification
- Timeline and important dates
- Financial information
- Legal and compliance matters
- Key events and decisions
- Risk identification
- Action items

### Caching and Performance
- SQLite-based caching
- Avoids regeneration of existing facts
- Automatic cleanup when collections are deleted
- Invalidation mechanism for document updates


## Prompt Templates

The fact extraction process uses three specialized prompt templates:

### `facts_extraction_initial.prompt`
Used for the first chunk of each file to establish the initial facts structure.

### `facts_extraction_update.prompt`
Used for subsequent chunks to iteratively extend the facts summary.

### `facts_finalization.prompt`
Used to create the final, well-organized executive summary with standardized sections.

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for LLM fact extraction (Together API key)
- `DOCUMENT_BASE_PATH`: Base path where SQLite database is created

### Runtime Dependencies
- Qdrant endpoint/index and API key must be configured for chunk retrieval
- Collection metadata is persisted in a SQLite database under the configured data directory

### Database Location
The SQLite database is created at:
```
{DOCUMENT_BASE_PATH}/collection_metadata.db
```

## Integration Points

### With VectorStoreManager
- Retrieves document chunks by collection
- Uses existing collection validation
- Leverages document metadata for ordering

### With Chainlit UI
- Progress callbacks for real-time updates
- Error toast notifications
- Loading message updates
- Formatted output display

### With Collection Management
- Automatic facts deletion when collections are removed
- Integration with collection statistics
- Cleanup when files are deleted

## Future Enhancements

- Incremental fact updates when new documents are added
- Export functionality for facts summaries
