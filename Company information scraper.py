import asyncio
import requests
import json
import pandas as pd
import time
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
LLM_MODEL = "llama3.1:70b" #Can replace with Qwen, Deepseek for chinese websites
LLM_URL = #Own API here
LLM_TIMEOUT = 300

# Keywords for link detection
KEYWORDS_EN = {
    "about": ["about", "company", "overview", "who-we-are", "profile", "about-us", 
              "company-profile", "our-story", "mission", "vision", "corporate", 
              "organization", "enterprise", "business"],
    "products": ["product", "service", "solution", "offering", "portfolio", 
                 "technology", "innovation", "capabilities", "expertise", 
                 "applications", "systems"]
}

KEYWORDS_CN = {
    "about": ["关于", "公司", "概述", "我们", "简介", "关于我们", 
              "公司简介", "企业介绍", "使命", "愿景", "企业", "组织", "机构"],
    "products": ["产品", "服务", "解决方案", "业务", "产品组合", "技术", 
                 "创新", "能力", "专业", "应用", "系统", "方案"]
}

ENGLISH_INDICATORS = ["english", "en", "eng", "英文", "英语版", "/en/", "/english/", "/en-us/", "/global/"]
CHINESE_DOMAINS = ['.cn', '.com.cn', 'baidu', 'tencent', 'alibaba', 'sina', 'qq']

# Data Models
class CompanyInfo(BaseModel):
    company_name: str = Field(..., description="Full official company name")
    sector: str = Field(..., description="Primary business sector")
    industry: str = Field(..., description="Specific industry within sector")
    products: List[str] = Field(default_factory=list, description="Products offered")
    services: List[str] = Field(default_factory=list, description="Services offered")
    technologies: List[str] = Field(default_factory=list, description="Key technologies")
    description: str = Field(..., description="Company description")
    market_focus: str = Field(default="", description="Primary market focus")

class ExtractionResult(BaseModel):
    success: bool
    ticker: str = ""
    url: str = ""
    company_info: Optional[Dict[str, Any]] = None
    scraped_urls: List[str] = []
    error: str = ""
    processing_time: float = 0.0

class FastScraper:
    """Optimized scraper with minimal Chinese detection overhead"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        })
    
    def is_chinese_domain(self, url: str) -> bool:
        """Quick Chinese domain detection"""
        return any(domain in url.lower() for domain in CHINESE_DOMAINS)
    
    def quick_language_check(self, text_sample: str) -> str:
        """Fast language detection using small text sample"""
        if len(text_sample) < 100:
            return "unknown"
        
        # Check first 500 chars only for speed
        sample = text_sample[:500]
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', sample))
        
        return "chinese" if chinese_chars > 20 else "english"
    
    def find_english_version_fast(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Fast English version detection - check obvious places only"""
        try:
            # Check only first 20 links to avoid slowdown
            for link in soup.find_all('a', href=True)[:20]:
                href = link.get('href', '').lower()
                text = link.get_text(strip=True).lower()
                
                # Check for obvious English indicators
                if any(ind in href for ind in ['/en/', '/english/', '/en-us/']):
                    return urljoin(base_url, link['href'])
                if any(ind in text for ind in ['english', 'en', '英文']):
                    return urljoin(base_url, link['href'])
            
            return None
        except Exception:
            return None
    
    def score_links_fast(self, soup: BeautifulSoup, base_url: str, keywords_en: List[str], keywords_cn: List[str]) -> Optional[str]:
        """Fast link scoring - check top candidates only"""
        best_score = 0
        best_url = None
        
        # Limit to first 30 links for speed
        for link in soup.find_all('a', href=True)[:30]:
            href = link.get('href', '').lower()
            text = link.get_text(strip=True).lower()
            
            score = 0
            
            # English keywords
            for keyword in keywords_en:
                if keyword in href:
                    score += 3
                elif keyword in text:
                    score += 2
            
            # Chinese keywords
            for keyword in keywords_cn:
                if keyword in href or keyword in text:
                    score += 3
            
            if score > best_score:
                best_score = score
                best_url = urljoin(base_url, link['href'])
                
                # Early exit if we find a perfect match
                if score >= 6:
                    break
        
        return best_url
    
    async def analyze_site_fast(self, url: str) -> Dict[str, Any]:
        """Fast site analysis with minimal overhead"""
        result = {
            "about": url,
            "products": url,
            "language": "english",  # Default assumption
            "english_version": None
        }
        
        try:
            # Single request to homepage
            response = self.session.get(url, timeout=10)  # Reduced timeout
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Quick language check on page title and first paragraph only
            title_text = soup.title.get_text() if soup.title else ""
            first_p = soup.find('p')
            first_p_text = first_p.get_text() if first_p else ""
            quick_sample = title_text + " " + first_p_text
            
            detected_lang = self.quick_language_check(quick_sample)
            result["language"] = detected_lang
            
            # Only look for English version if we detect Chinese AND it's a Chinese domain
            if detected_lang == "chinese" and self.is_chinese_domain(url):
                english_url = self.find_english_version_fast(soup, url)
                if english_url and english_url != url:
                    try:
                        eng_response = self.session.get(english_url, timeout=8)
                        eng_response.raise_for_status()
                        soup = BeautifulSoup(eng_response.text, 'lxml')
                        result["language"] = "english"
                        result["english_version"] = english_url
                        url = english_url  # Use English version
                        logger.info(f"Using English version: {english_url}")
                    except Exception:
                        pass  # Continue with Chinese version
            
            # Find relevant pages (fast scoring)
            about_url = self.score_links_fast(soup, url, KEYWORDS_EN["about"], KEYWORDS_CN["about"])
            products_url = self.score_links_fast(soup, url, KEYWORDS_EN["products"], KEYWORDS_CN["products"])
            
            result["about"] = about_url or url
            result["products"] = products_url or url
            
        except Exception as e:
            logger.warning(f"Fast analysis failed for {url}: {e}")
        
        return result
    
    async def scrape_content_fast(self, url: str) -> Tuple[bool, str]:
        """Fast content scraping with optimized crawl4ai settings"""
        is_chinese = self.is_chinese_domain(url)
        
        browser_config = BrowserConfig(
            verbose=False,
            headless=True,
            java_script_enabled=True,
            user_agent=self.session.headers['User-Agent']
        )
        
        # Optimized timeouts
        run_config = CrawlerRunConfig(
            word_count_threshold=1,
            cache_mode=CacheMode.BYPASS,
            delay_before_return_html=1.5,  # Reduced delay
            page_timeout=25000 if is_chinese else 20000,  # Reduced timeouts
            js_code=[
                "window.scrollTo(0, document.body.scrollHeight/3);",  # Less scrolling
                "await new Promise(resolve => setTimeout(resolve, 1000));"  # Reduced wait
            ]
        )
        
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=run_config),
                    timeout=35  # Reduced total timeout
                )
                
                if result.success and result.cleaned_html:
                    return True, result.cleaned_html
                else:
                    raise Exception("Crawl4ai extraction failed")
        
        except Exception:
            # Quick fallback to requests
            try:
                response = self.session.get(url, timeout=8)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks if chunk)
                
                return bool(clean_text), clean_text
            except Exception:
                return False, ""

class OptimizedLLMExtractor:
    """LLM extractor with larger context and better prompts"""
    
    def __init__(self):
        self.model = LLM_MODEL
        self.url = LLM_URL
        self.timeout = LLM_TIMEOUT
    
    def create_comprehensive_prompt(self, content: str, url: str, language: str) -> str:
        """Enhanced prompt for comprehensive extraction"""
        # Use more content - 25KB instead of 15KB
        content_sample = content[:15000]
        
        base_prompt = f"""
You are an expert at extracting comprehensive company information from web content. 

Website URL: {url}
Website Content:
{content_sample}

Extract ALL available information and return as a valid JSON object.

{{
    "company_name": "Full official company name",
    "sector": "Primary business sector (e.g., Healthcare, Technology, Manufacturing, Energy, Finance)",
    "industry": "Specific industry within the sector (e.g., Medical Devices, Cloud Software, Renewable Energy)",
    "products": ["COMPLETE list of ALL products mentioned - be exhaustive, include product names, model numbers, variants"],
    "services": ["COMPLETE list of ALL services mentioned - include consulting, support, maintenance, etc."],
    "technologies": ["COMPLETE list of ALL technologies, platforms, software, methodologies mentioned"],
    "description": "COMPREHENSIVE description including company mission, history, business model, and key differentiators",
    "market_focus": "Detailed description of target markets, customer segments, geographic focus, and market position"
}}

CRITICAL INSTRUCTIONS:
1. Be EXHAUSTIVE - extract EVERY product, service, and technology mentioned
2. Include specific product names, model numbers, service types, and technical details
3. If you see lists or bullet points, extract ALL items
4. For Chinese content, translate everything to English but preserve original Chinese names in parentheses
5. Never truncate lists - include everything you find
6. If information spans multiple sections, combine all relevant details
7. Return ONLY the JSON object, no additional text and valid syntax
"""
        
        if language == "chinese":
            base_prompt += """

CHINESE CONTENT HANDLING:
- Translate all content to English for consistency
- For products/services, provide English translations with Chinese originals when relevant
- Maintain technical accuracy in translations
"""
        
        return base_prompt + "\nReturn only the complete JSON object with ALL available information."
    
    def query_llm_optimized(self, prompt: str) -> str:
        """Optimized LLM query with better parameters"""
        try:
            data = {
                "prompt": prompt,
                "model": self.model,
                "stream": False,
                "options": {
                    "num_ctx": 16384, 
                    "temperature": 0.2,  # Lower temperature for more consistent extraction
                    "top_p": 0.3,
                    "top_k": 40,
                    "max_new_tokens": 2000,  
                },
            }
            
            response = requests.post(self.url, json=data, stream=False, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["response"]
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def parse_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Find JSON in response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON found in LLM response")
            
            json_str = llm_response[json_start:json_end]
            company_info = json.loads(json_str)
            
            # Validate using Pydantic
            validated_info = CompanyInfo(**company_info)
            return validated_info.model_dump()
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise

class OptimizedCompanyExtractor:
    """Main extractor with parallel processing and optimizations"""
    
    def __init__(self):
        self.scraper = FastScraper()
        self.llm = OptimizedLLMExtractor()
    
    async def extract_company_info(self, url: str) -> ExtractionResult:
        """Optimized extraction with parallel processing"""
        start_time = time.time()
        logger.info(f"Processing: {url}")
        
        try:
            # Step 1: Fast site analysis
            site_info = await self.scraper.analyze_site_fast(url)
            detected_language = site_info["language"]
            
            logger.info(f"Language: {detected_language}, About: {site_info['about']}, Products: {site_info['products']}")
            
            # Step 2: Parallel content scraping
            scraping_tasks = []
            urls_to_scrape = []
            
            # Always scrape both about and products for comprehensiveness
            for page_type in ["about", "products"]:
                page_url = site_info[page_type]
                if page_url and page_url not in urls_to_scrape:
                    urls_to_scrape.append(page_url)
                    scraping_tasks.append(self.scraper.scrape_content_fast(page_url))
            
            # Execute scraping in parallel
            logger.info(f"Scraping {len(scraping_tasks)} pages in parallel...")
            scraping_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            
            # Collect successful content
            all_content = []
            scraped_urls = []
            
            for i, result in enumerate(scraping_results):
                if isinstance(result, Exception):
                    logger.warning(f"Scraping failed for {urls_to_scrape[i]}: {result}")
                    continue
                
                success, content = result
                if success and content:
                    page_type = "ABOUT" if i == 0 else "PRODUCTS"
                    all_content.append(f"=== {page_type} PAGE ===\n{content}")
                    scraped_urls.append(urls_to_scrape[i])
            
            if not all_content:
                return ExtractionResult(
                    success=True,
                    url=url,
                    company_info=company_info,
                    
                    scraped_urls=scraped_urls,
                    processing_time=time.time() - start_time
                )
            
            # Step 3: LLM extraction with comprehensive prompt
            combined_content = "\n\n".join(all_content)
            prompt = self.llm.create_comprehensive_prompt(combined_content, url, detected_language)
            
            logger.info("Extracting comprehensive information with LLM...")
            llm_response = self.llm.query_llm_optimized(prompt)
            company_info = self.llm.parse_response(llm_response)
            
            # Add metadata
            company_info["original_language"] = detected_language
            
            return ExtractionResult(
                success=True,
                url=url,
                company_info=company_info,
                detected_language=detected_language,
                english_version_used=site_info.get("english_version") is not None,
                scraped_urls=scraped_urls,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Extraction failed for {url}: {e}")
            return ExtractionResult(
                success=False,
                url=url,
                error=str(e),
                processing_time=time.time() - start_time
            )

class CSVProcessor:
    """Optimized CSV processor"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.extractor = OptimizedCompanyExtractor()
    
    def detect_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Auto-detect ticker and URL columns"""
        ticker_col = None
        url_col = None
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['ticker', 'symbol', 'code']):
                ticker_col = col
                break
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['url', 'link', 'website', 'site']):
                url_col = col
                break
        
        if not ticker_col:
            ticker_col = df.columns[0]
            logger.warning(f"Using '{ticker_col}' as ticker column")
        
        if not url_col:
            url_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            logger.warning(f"Using '{url_col}' as URL column")
        
        return ticker_col, url_col
    
    async def process_companies(self, start_from: int = 0) -> List[ExtractionResult]:
        """Process companies with optimized settings"""
        df = pd.read_csv(self.csv_file)
        ticker_col, url_col = self.detect_columns(df)
        
        logger.info(f"Processing {len(df)} companies from {self.csv_file}")
        logger.info(f"Columns: ticker='{ticker_col}', url='{url_col}'")
        
        results = []
        start_time = time.time()
        
        for idx, row in df.iloc[start_from:].iterrows():
            ticker = str(row.get(ticker_col, f'Company_{idx}')).strip()
            company_url = str(row.get(url_col, '')).strip()
            
            if not company_url or company_url.lower() in ['nan', 'none', '']:
                results.append(ExtractionResult(
                    success=False,
                    ticker=ticker,
                    url="",
                    error="No valid URL provided"
                ))
                continue
            
            if not company_url.startswith(('http://', 'https://')):
                company_url = 'https://' + company_url
            
            # Progress tracking
            current_idx = len(results) + start_from
            elapsed = time.time() - start_time
            if len(results) > 0:
                avg_time = elapsed / len(results)
                remaining = len(df) - current_idx
                eta = avg_time * remaining / 60
                logger.info(f"Progress: {current_idx+1}/{len(df)} | Avg: {avg_time:.1f}s | ETA: {eta:.1f}min")
            
            try:
                result = await asyncio.wait_for(
                    self.extractor.extract_company_info(company_url),
                    timeout=120  # Reduced timeout due to optimizations
                )
                result.ticker = ticker
                results.append(result)
                
                if result.success:
                    logger.info(f"✅ {ticker}: {result.company_info.get('company_name', 'Unknown')} ({result.processing_time:.1f}s)")
                else:
                    logger.error(f"❌ {ticker}: {result.error}")
                
            except asyncio.TimeoutError:
                logger.error(f"❌ {ticker}: Timeout")
                results.append(ExtractionResult(
                    success=False,
                    ticker=ticker,
                    url=company_url,
                    error="Processing timeout"
                ))
            
            # Save progress
            if len(results) % 5 == 0:
                self.save_results(results, f"temp_results_{current_idx}.json")
            
            # Reduced rate limiting since we're faster now
            await asyncio.sleep(1.5)
        
        return results
    
    def save_results(self, results: List[ExtractionResult], filename: str = "company_extraction_results.json"):
        """Save results in JSON and CSV formats"""
        try:
            results_dict = [result.model_dump() for result in results]
            
            # Save JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            # Save comprehensive CSV
            csv_filename = filename.replace('.json', '.csv')
            csv_data = []
            
            for result in results:
                if result.success and result.company_info:
                    info = result.company_info
                    csv_data.append({
                        'ticker': result.ticker,
                        'company_name': info.get('company_name', ''),
                        # Remove: 'company_name_chinese': info.get('company_name_chinese', ''),
                        'sector': info.get('sector', ''),
                        'industry': info.get('industry', ''),
                        'products': '; '.join(info.get('products', [])),
                        'services': '; '.join(info.get('services', [])),
                        'technologies': '; '.join(info.get('technologies', [])),
                        'description': info.get('description', ''),
                        'market_focus': info.get('market_focus', ''),
                        # Remove: 'original_language': info.get('original_language', 'unknown'),
                        # Remove: 'detected_language': result.detected_language,
                        # Remove: 'english_version_used': result.english_version_used,
                        'processing_time': f"{result.processing_time:.1f}s",
                        'scraped_urls': '; '.join(result.scraped_urls),
                        'url': result.url,
                        'extraction_status': 'SUCCESS'
                    })
                else:
                    csv_data.append({
                        'ticker': result.ticker,
                        'company_name': '',
                        # Remove: 'company_name_chinese': '',
                        'sector': '',
                        'industry': '',
                        'products': '',
                        'services': '',
                        'technologies': '',
                        'description': '',
                        'market_focus': '',
                        # Remove: 'original_language': '',
                        # Remove: 'detected_language': result.detected_language,
                        # Remove: 'english_version_used': False,
                        'processing_time': f"{result.processing_time:.1f}s",
                        'scraped_urls': '',
                        'url': result.url,
                        'extraction_status': f"FAILED: {result.error}"
                    })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            logger.info(f"✅ Results saved: JSON={filename}, CSV={csv_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_comprehensive_summary(self, results: List[ExtractionResult]):
        """Print detailed summary with complete information"""
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.processing_time for r in results)
        
        print(f"\n{'='*100}")
        print(f"📊 COMPREHENSIVE EXTRACTION SUMMARY")
        print(f"{'='*100}")
        print(f"Total companies: {len(results)}")
        print(f"Successful extractions: {successful}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per company: {total_time/len(results):.1f} seconds")
        
        # Show complete successful extractions
        for result in results:
            if result.success and result.company_info:
                info = result.company_info
      
                print(f"\n{'='*100}")
                print(f"🏢 {info.get('company_name', 'Unknown')} ({result.ticker})")
                print(f"⏱️  Processing time: {result.processing_time:.1f}s")
                print(f"🏭 Sector: {info.get('sector', 'N/A')}")
                print(f"🔧 Industry: {info.get('industry', 'N/A')}")
                print(f"🎯 Market Focus: {info.get('market_focus', 'N/A')}")
                print(f"\n📄 Description:\n{info.get('description', 'N/A')}")
                
                # Show ALL products, services, technologies - no truncation
                if info.get('products'):
                    print(f"\n📦 Products ({len(info['products'])} total):")
                    for i, product in enumerate(info['products'], 1):
                        print(f"   {i}. {product}")
                
                if info.get('services'):
                    print(f"\n🔧 Services ({len(info['services'])} total):")
                    for i, service in enumerate(info['services'], 1):
                        print(f"   {i}. {service}")
                
                if info.get('technologies'):
                    print(f"\n💻 Technologies ({len(info['technologies'])} total):")
                    for i, tech in enumerate(info['technologies'], 1):
                        print(f"   {i}. {tech}")

async def main():
    """Optimized main execution"""
    # Update this path to your CSV file
    csv_file = "/Users/yilanliu/Downloads/rivermap internship/csv and excels/company_links for scraping copy.csv"
    
    try:
        processor = CSVProcessor(csv_file)
        
        logger.info("🚀 Starting optimized company extraction...")
        results = await processor.process_companies()
        
        # Save comprehensive results
        processor.save_results(results)
        
        # Print complete summary
        processor.print_comprehensive_summary(results)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"🎉 Extraction complete! {successful}/{len(results)} companies processed successfully")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
