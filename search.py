from bs4 import BeautifulSoup
import requests
import pandas as pd
import googleapiclient.errors
from googleapiclient.discovery import build
import googleapiclient.errors
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from requests.exceptions import SSLError
from urllib.parse import urlparse
import re
from nltk.tokenize import sent_tokenize
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--headless")  # Enable headless mode
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
chrome_options.add_argument("--disable-gpu")  # Applicable to Windows
chrome_options.add_argument("--window-size=1920x1080")  # Set the window size (optional)



sites_source = ["www.ellinikahoaxes.gr","factcheckgreek.afp.com","www.check4facts.gr",
                "factcheckcyprus.org",'www.youtube.com','www.linkedin.com', "m.facebook.com"]
doc_extensions = ["doc", "docx", 'php', 'pdf', 'txt', 'theFile', 'file', 'xls']

class GoogleSearch:

    def __init__(self, engine_url):
        self.engine_url = engine_url
        self.driver = webdriver.Chrome(options=chrome_options)
        # self.driver = webdriver.Chrome()
        

        

    def google_search(self, text, total_num):   
        max_iter=0
        urls = set()
        pattern = r'[./=]([a-zA-Z0-9]+)$'
        
        self.driver.get(f"{self.engine_url}/search?q={text}")
        self.driver.implicitly_wait(2)  
        time.sleep(2)
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        

        try:
            accept_all_button = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//div[@role='dialog']//button[@id='L2AGLb']")))
            accept_all_button.click()
        except:
            print("No 'Accept All' button found or failed to click.")
            
        
      
        
        
        while(len(urls)<total_num):

            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            search = soup.find_all('div', class_="yuRUbf")
            #if search is empty, modify the string accordingly
            if (not search):
                print('tokenizing claim....')
                self.driver.get(f"{self.engine_url}/search?q={sent_tokenize(text)[0]}")
                self.driver.implicitly_wait(10)  
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                search = soup.find_all('div', class_="yuRUbf")
               
                if not search:
                    print('comma seperating claim....')
                    self.driver.get(f"{self.engine_url}/search?q={text.split(',')[0]}")
                    self.driver.implicitly_wait(10)  
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    search = soup.find_all('div', class_="yuRUbf")
                    


            for h in search:
                
                #if the url is valid continue, else move on to the next one
                try:
                    response = requests.get(h.a.get('href'), timeout=10)
                except SSLError as e:
                    print("SSL Error:", e)
                    print('On url: ', h.a.get('href'))
                    continue
                except TimeoutError as e:
                    continue
                except Exception as e:
                    print("Exception Error:", e)
                    print('On url: ', h.a.get('href'))
                    continue


                #if the url actually has content
                if(response.content):
                    #if url is not a file of some sort
                    match = re.search(pattern, h.a.get('href')[-6:])
                    if match:
                        file_extension = match.group(1)
                    else:
                        file_extension = None
                    #file_extension = h.a.get('href').lower().split('.')[-1]
                    
                    url_domain = urlparse(h.a.get('href')).netloc
                    # print(url_domain)
                    if file_extension not in doc_extensions and url_domain not in sites_source and not '/document/' in h.a.get('href'):
                        #print(h.a.get('href'))
                        urls.add(h.a.get('href'))
                        
                    else:
                        continue
                else:
                    continue

            
            #if we still havent found the desired url number, scroll down on the driver           
            if(len(urls)<total_num):
                print('Scrolling down...')
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                #if we reach the end of the page, harvest the rest of the urls
                if last_height == new_height and 0<len(urls)<total_num:
                    print('end of page reached')
                    max_iter+=1
                #if not even one url was found, split the claim in half
                elif len(urls)==0:
                    print('splitting claim')
                    text = text[:len(text)//2]
                    self.driver.get(f"{self.engine_url}/search?q={text}")    
                else:
                    last_height = new_height

            #if even after the end of the page the total number of urls is not met, 
            if max_iter>1:   
                print('reducing the number of urls......')
                total_num = total_num -1
                #break
                

                

        



   

        self.driver.quit()

        
 
        return self.remove_duplicate_urls(urls)[:total_num]

            

    def remove_duplicate_urls(self, urls):
        unique_domains = set()
        cleaned_urls = []


        #remove duplicate urls
        for url in urls:
            # Parse the URL to get the domain
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            if domain not in unique_domains:
                unique_domains.add(domain)
                cleaned_urls.append(url)



        return cleaned_urls








