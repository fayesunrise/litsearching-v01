

# Literature Searching Tool

An academic research tool built with Streamlit. This assistant leverages the **Semantic Scholar Graph API** for discovery and **Google Gemini 2.0 Flash** for RAG-based (Retrieval-Augmented Generation) synthesis and analysis.

## Key Features

* **AI Research Insights**: 
    * **Synthesis**: Automatically generates a 5–10 sentence thematic summary with [P#] citations.
    * **Controversies & Inconsistencies**: A specialized section identifying conflicting findings, methodological differences, or unresolved debates.
    * **Keyword Extraction**: Identifies 8–12 research-useful terms.
* **Visual Analytics**: 
    * **Publication Trends**: Interactive bar chart (powered by Altair) showing the distribution of search results by year.
    * **Venue Analysis**: Breakdown of top journals and conferences for the current search.
* **Intelligent Chat**: A Scholar-lab style interface to ask specific questions about your results, grounded strictly in the provided paper context.
* **Research Library**: Personal session-based "Saved Papers" list to curate a custom corpus for AI analysis.
* **Dynamic Context Syncing**: AI synthesis automatically scales to analyze the exact number of results selected (up to 50 papers simultaneously).


## Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/fayesunrise/litsearching-v01.git](https://github.com/fayesunrise/litsearching-v01.git)
   cd litsearching-v01
