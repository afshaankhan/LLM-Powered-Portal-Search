import os
import json
import scrapy
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager  # Optional: to automatically manage driver
from scrapy.http import HtmlResponse
import time
from selenium.webdriver.edge.options import Options

class TejasSpiderSpider(scrapy.Spider):
    name = "tejas_spider"
    # allowed_domains = ["irb.co.in"]
    # start_urls = ["https://www.irb.co.in"]
    allowed_domains = ["orientindia.in"]
    start_urls = ["https://www.orientindia.in"]
    res_dir = 'files'
    media_file_extensions = ['.jpg', '.png', 'jpeg', '.gif', '.mp4', '.pdf', '.doc', '.docx', '.xlsx', '.xlsm', '.xls', '.mp3', '.ashx', '.ppt', '.pptx', '.txt', '.zip', '.rar', '.7z', '.tar', '.gz', '.tgz', '.exe', '.msi', '.apk', '.ipa', '.dmg', '.pkg', '.deb', '.rpm', '.iso', '.img', '.bin', '.cue', '.mdf', '.nrg', '.vcd', '.avi', '.mkv', '.flv', '.mov', '.wmv', '.mpg', '.mpeg', '.m4v', '.webm', '.vob', '.3gp', '.3g2', '.m2v', '.m4v', '.f4v', '.f4p',]

    def __init__(self, *args, **kwargs):
        super(TejasSpiderSpider, self).__init__(*args, **kwargs)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        res_file_name = self.start_urls[0].split('//')[1].replace('www.', '').replace('.com', '').replace('/','-') + '.json'
        self.scrape_res_file = os.path.join(self.res_dir, res_file_name)
        self.is_scraped = {}
        self.is_scraped[self.start_urls[0]] = True
        self.scrape_res = {}

        options = Options()
        options.add_argument("--headless")  # Enable headless mode
        # Initialize Edge WebDriver
        self.driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=options)
        # self.driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()))

    def remove_empty_strings(self, raw_text):
        for i in range(len(raw_text)-1, -1, -1):
            if raw_text[i].strip() == '':
                del raw_text[i]
                continue   
        return raw_text

    def remove_duplicates(self, raw_text):
        return list(set(raw_text))
    
    def remove_subset_text(self, raw_text):
        i = 0
        while i<len(raw_text):
            j = i+1
            while j<len(raw_text):
                if raw_text[i] in raw_text[j]:
                    del raw_text[i]
                    j += 1
                elif raw_text[j] in raw_text[i]:
                    del raw_text[j]
                else:
                    j += 1
            i += 1
        return raw_text

    def remove_consecutive_spaces(self, raw_text):
        for i in range(len(raw_text)-1, -1, -1):
            raw_text[i] = ' '.join(raw_text[i].split()).strip()
        return raw_text
    
    def clean_raw_text(self, raw_text):
        raw_text = self.remove_empty_strings(raw_text)
        raw_text = self.remove_duplicates(raw_text)
        raw_text = self.remove_subset_text(raw_text)
        raw_text = self.remove_consecutive_spaces(raw_text)
        return raw_text

    def filter_hrefs(self, hrefs, response):
        hrefs = [response.urljoin(x) for x in hrefs]
        hrefs = [x for x in hrefs if self.start_urls[0] in x]
        filtered_hrefs = []
        for href in hrefs:
            ends = False
            for ext in self.media_file_extensions:
                if href.lower().endswith(ext):
                    ends = True
                    break
            if not ends:
                filtered_hrefs.append(href)
        filtered_hrefs = list(set(filtered_hrefs))
        return filtered_hrefs

    def parse(self, response):
        print(f'No of urls scraped: {len(self.scrape_res)}')
        print(f'Parsing URL: {response.url}')

        # Use Edge to fetch the page
        self.driver.get(response.url)
        time.sleep(5)  # Wait for JavaScript to load; adjust the delay as needed
        
        # Capture page content and create a new Scrapy response
        page_content = self.driver.page_source
        selenium_response = HtmlResponse(url=response.url, body=page_content, encoding='utf-8')

        try:
            all_text = selenium_response.xpath('//body//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::noscript)]').getall()
        except:
            return
        
        all_text = self.clean_raw_text(all_text)
        self.scrape_res[response.url] = '\n'.join(all_text)

        hrefs = selenium_response.css('a::attr(href)').getall()
        hrefs = self.filter_hrefs(hrefs, response)
        print(f'{hrefs=}')
        with open(self.scrape_res_file, 'w', encoding='utf-8') as text_file:
            json.dump(self.scrape_res, text_file, indent=4)
        for i in range(len(hrefs)):
            parsed = self.is_scraped.get(hrefs[i], False)
            if parsed:
                continue
            self.is_scraped[hrefs[i]] = True
            # yield response.follow(hrefs[i], callback=self.parse)
            yield scrapy.Request(response.urljoin(hrefs[i]), callback=self.parse)
        return
    
    def closed(self, reason):
        # Save the URL to text content dictionary to JSON
        with open(self.scrape_res_file, 'w', encoding='utf-8') as text_file:
            json.dump(self.scrape_res, text_file, indent=4)
        return