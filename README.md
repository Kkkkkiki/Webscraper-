# Company Information WebScraper

Herro! 
This is an advanced web scraping system that extracts comprehensive company information from corporate websites using AI-powered content analysis. Built with Python, featuring async processing, multi-language support, and LLM integration for intelligent data extraction.

## 🚀 Key Features

- **AI-Powered Extraction**: Uses Llama 3.1 LLM for intelligent content analysis and structured data extraction
- **Multi-Language Support**: Automatically detects Chinese/English content and handles language switching
- **Async Processing**: High-performance parallel scraping with optimized timeouts and rate limiting
- **Smart Navigation**: Automatically discovers and navigates to About/Products pages using keyword scoring
- **Comprehensive Data Model**: Extracts company name, sector, industry, products, services, technologies, and market focus
- **Robust Error Handling**: Graceful fallbacks, retry mechanisms, and detailed logging
- **CSV Batch Processing**: Process multiple companies from CSV files with progress tracking
- **Multiple Output Formats**: JSON and CSV export with comprehensive results

## 📋 Prerequisites

- Python 3.8+
- Ollama server running locally with Llama 3.1:70b model
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/webscraper.git
cd webscraper
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Ollama and your own API**
```

## 📁 Project Structure

```
webscraper/
│
├── Company information scraper.py  # Main scraper implementation
├── requirements.txt               # Python dependencies
├── company_links.csv             # Input CSV with company URLs
├── results/
│   ├── company_extraction_results.json
│   ├── company_extraction_results.csv
│   └── temp_results_*.json       # Progress backup files
├── logs/                         # Automatic logging
└── README.md
```

## 🚀 Quick Start

### Single Company Extraction

```python
import asyncio
from Company_information_scraper import OptimizedCompanyExtractor

async def extract_single_company():
    extractor = OptimizedCompanyExtractor()
    result = await extractor.extract_company_info("https://example-company.com")
    
    if result.success:
        print(f"Company: {result.company_info['company_name']}")
        print(f"Sector: {result.company_info['sector']}")
        print(f"Products: {result.company_info['products']}")
    else:
        print(f"Extraction failed: {result.error}")

asyncio.run(extract_single_company())
```

### Batch Processing from CSV

```python
import asyncio
from Company_information_scraper import CSVProcessor

async def process_csv():
    processor = CSVProcessor("company_links.csv")
    results = await processor.process_companies()
    processor.save_results(results)
    processor.print_comprehensive_summary(results)

asyncio.run(process_csv())
```

### CSV Format Requirements

Your CSV file should contain columns for company identifiers and URLs:

```csv
ticker,url
AAPL,https://apple.com
MSFT,https://microsoft.com
GOOGL,https://alphabet.com
```

The system auto-detects column names containing: `ticker`, `symbol`, `code` for identifiers and `url`, `link`, `website` for URLs.

### Language Detection

The system automatically:
- Detects Chinese domains (`.cn`, `baidu`, `tencent`, etc.)
- Analyzes content language using character patterns
- Searches for English versions on Chinese sites
- Switches to English content when available

### Content Extraction Strategy

1. **Site Analysis**: Discovers About/Products pages using keyword scoring
2. **Parallel Scraping**: Simultaneous extraction from multiple relevant pages
3. **Content Processing**: Advanced cleaning and text preparation
4. **LLM Analysis**: AI-powered structured data extraction

## 📊 Extracted Data Structure

```json
{
  "company_name": "Apple Inc.",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "products": [
    "iPhone 15 Pro",
    "MacBook Air M2",
    "iPad Pro",
    "Apple Watch Series 9"
  ],
  "services": [
    "iCloud Storage",
    "Apple Care",
    "App Store",
    "Apple Music"
  ],
  "technologies": [
    "M2 Chip",
    "iOS 17",
    "Metal 3",
    "Neural Engine"
  ],
  "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
  "market_focus": "Premium consumer electronics, digital services, and ecosystem integration"
}
```


## 📈 Performance Optimizations

- **Async Processing**: Parallel page scraping
- **Smart Timeouts**: Different timeouts for Chinese vs English sites
- **Content Sampling**: Uses 15KB samples for LLM analysis
- **Rate Limiting**: Respectful 1.5s delays between requests
- **Progress Saving**: Auto-saves every 5 companies

## 📝 Output Files

### JSON Results
Comprehensive extraction results with full metadata:
```
company_extraction_results.json
```

### CSV Results
Tabular format for analysis:
```
company_extraction_results.csv
```

### Temporary Files
Progress backups created every 5 companies:
```
temp_results_15.json
temp_results_20.json
```

## 🔧 Dependencies

```txt
asyncio
requests
pandas
beautifulsoup4
crawl4ai
pydantic
lxml
ollama (server)
```

## 🚨 Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

**Slow Extraction:**
```python
# Reduce model size for faster processing
LLM_MODEL = "llama3.1:8b"  # Instead of 70b
```

**Memory Issues:**
```python
# Reduce content sample size
content_sample = content[:10000]  # Instead of 15000
```

### Performance Tuning

- **For Speed**: Use smaller LLM models (8b instead of 70b)
- **For Accuracy**: Increase content sample size and timeout values
- **For Scale**: Adjust rate limiting and parallel processing limits

