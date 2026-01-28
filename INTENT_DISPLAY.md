# Intent Display Feature

## Overview

The chatbot now displays the detected intent and confidence score before each response, providing transparency about how the query was interpreted.

## Display Format

### Console Application (C#)

When using the C# console application, the intent appears in cyan color before the response:

```
You: How do I configure the database connection?

Processing...

🎯 Intent: How To | Confidence: 87.3%