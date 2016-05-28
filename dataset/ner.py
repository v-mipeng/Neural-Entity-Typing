import time
import codecs


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from Tkinter import Tk

r = Tk()
r.withdraw()

driver = webdriver.Chrome(r"D:\Software Install\Chrome Driver\chromedriver.exe")
driver.get("https://www.microsoft.com/cognitive-services/en-us/entity-linking-intelligence-service")
text_box = driver.find_element_by_name("Text")
link_button = driver.find_element_by_id("ButtonBreak")
json_button = driver.find_element_by_xpath("//li[@id='codeNav']/a[1]/span")
json_box = driver.find_element_by_id("jsonOutput")
json_button.click()

def ner(doc):
    text_box.clear()
    text_box.send_keys(doc)
    link_button.click()
    time.sleep(0.5)
    while True:
        try:
            json_button.click()
            break
        except:
            time.sleep()
    json_doc = driver.find_element_by_id("jsonOutput").text
    return json_doc
