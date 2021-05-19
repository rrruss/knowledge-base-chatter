import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime

import boto3
import pandas as pd
import scrapy
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerRunner
from scrapy.exceptions import IgnoreRequest
from scrapy.spidermiddlewares.httperror import HttpError
from scrapy.utils.log import configure_logging
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks
from twisted.internet.error import DNSLookupError
from twisted.internet.task import LoopingCall

LOCAL_OUTPUT_PATH = "scraped_text.csv"

USER_AGENTS = [
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/57.0.2987.110 "
        "Safari/537.36"
    ),  # chrome
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/61.0.3163.79 "
        "Safari/537.36"
    ),  # chrome
    (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) "
        "Gecko/20100101 "
        "Firefox/55.0"
    ),  # firefox
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/61.0.3163.91 "
        "Safari/537.36"
    ),  # chrome
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/62.0.3202.89 "
        "Safari/537.36"
    ),  # chrome
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/63.0.3239.108 "
        "Safari/537.36"
    ),  # chrome
]


class Page(scrapy.Item):
    url = scrapy.Field()
    description = scrapy.Field()
    body = scrapy.Field()


class PageSpider(scrapy.Spider):
    name = "pagescrape"
    custom_settings = {
        # Fast recommendations:
        "SCHEDULER_PRIORITY_QUEUE": "scrapy.pqueues.DownloaderAwarePriorityQueue",
        "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleFifoDiskQueue",
        "SCHEDULER_MEMORY_QUEUE": "scrapy.squeues.FifoMemoryQueue",
        "REACTOR_THREADPOOL_MAXSIZE": 1000,
        "COOKIES_ENABLED": False,
        "RETRY_ENABLED": False,
        "DOWNLOAD_TIMEOUT": 8,
        "REDIRECT_ENABLED": True,
        "AJAXCRAWL_ENABLED": True,
        "DEPTH_PRIORITY": 1,
        "CONCURRENT_REQUESTS": 1000,
        "DNSCACHE_ENABLED": False,
        # Rotate user agents
        "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
        "scrapy_useragents.downloadermiddlewares.useragents.UserAgentsMiddleware": 500,
        # logging.
        "LOG_ENABLED": True,
        "LOG_LEVEL": logging.INFO,
        "AUTOTHROTTLE_ENABLED": False,
        # Honoring 1 second delay per page, and adhere to robots.txt.
        "DOWNLOAD_DELAY": 1,
        "ROBOTSTXT_OBEY": True,
        # Export to s3.
        "FEED_URI": LOCAL_OUTPUT_PATH,
        "FEED_EXPORT_FIELDS": ["url", "description", "body"],
        "FEED_FORMAT": "csv",
    }

    def __init__(self, urls):
        self.urls = urls

    def start_requests(self):
        self.crawler.stats.set_value("urls_count", len(self.urls))
        for url in self.urls:
            yield scrapy.Request(
                url=url, callback=self.parse, errback=self.request_errback
            )

    def parse_metatags(self, response):
        description = " ".join(
            response.xpath("//meta[@name='description']/@content").getall()
        ).strip()
        keywords = " ".join(
            response.xpath("//meta[@name='keywords']/@content").getall()
        ).strip()
        title = " ".join(response.css("title::text").getall()).strip()
        meta = " ".join([description, keywords, title])
        return meta

    def parse_body(self, response):
        text = BeautifulSoup(response.body, "lxml").get_text(" ", strip=True)
        return text

    def parse(self, response):
        meta = self.parse_metatags(response)
        text = self.parse_body(response)
        if meta or text:
            page = Page()
            page["url"] = response.meta.get("url")
            page["description"] = meta or ""
            page["body"] = text or ""
            yield page
        else:
            self.logger.debug(f"No meta or text for {response.request.url}")

    def request_errback(self, failure):
        url = failure.request.url
        if failure.check(HttpError):
            self.logger.info(f"HttpError in {url}")
        elif failure.check(DNSLookupError):
            self.logger.info(f"DNSLookupError in {url}")
        elif failure.check(IgnoreRequest):
            self.logger.info(f"IgnoreRequest in {url}")
        else:
            failure.printTraceback()


def get_urls_to_scrape():
    return [
        "https://slack.com/slack-tips/share-code-snippets",
        "https://slack.com/slack-tips/let-your-team-know-your-working-hours",
    ]


@inlineCallbacks
def scrape_sites(urls):
    # If a previous run failed before deleting the output file we
    # accidentally up appending to existing content, unless we delete
    # any existing file.
    if os.path.exists(LOCAL_OUTPUT_PATH):
        os.remove(LOCAL_OUTPUT_PATH)

    print(f"Crawling {len(urls)} urls")
    start = time.time()
    runner = CrawlerRunner()
    crawler = runner.create_crawler(PageSpider)
    # Runner.crawl immediately returns a Deferred object before scraping has taken place.
    # Runner.crawl simple schedules some crawling to run as soon as possible in reactors event loop.
    # In general with Deferred objects, you can attach at callback to them which will fire when the
    # event loop has finished processing the corresponding work. However, using the @inlineCallbacks
    # decorator we can block the program here using yield until the scraping has been completed to
    # make the code execution more intuitive.
    yield runner.crawl(crawler, urls=urls)
    print(
        "Done processing {} urls in {:.2f} seconds".format(
            len(urls), time.time() - start
        )
    )

    # Load scrapy's data and remove local file
    result_df = pd.read_csv(LOCAL_OUTPUT_PATH, names=["url", "description", "body"])
    os.remove(LOCAL_OUTPUT_PATH)

    return result_df


def upload_to_s3(filename, result_df):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")

    s3_path = f"scraped_data/slack/date={date}/{filename}"
    s3 = boto3.client("s3")
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, header=False, index=False)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket="kbchatter", Key=s3_path)
    print(f"Uploaded output to {s3_path}")


@inlineCallbacks
def main(upload):
    urls = get_urls_to_scrape()
    result_df = yield scrape_sites(urls)
    if upload:
        upload_to_s3(filename="test", result_df=result_df)
    else:
        print(result_df.head(n=50))
    reactor.stop()


def loop_failed(failure, loop):
    print("LoopingCall failed. Shutting down reactor")
    print(failure)
    reactor.stop()
    print("loop_failed end")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape URLs")
    parser.add_argument(
        "--upload",
        dest="upload",
        action="store_true",
        help="Upload scraped data to S3.",
    )
    parser.add_argument(
        "--no-upload",
        dest="upload",
        action="store_false",
        help="Scrape data without uploading to S3. First 50 results will be displayed.",
    )
    parser.set_defaults(upload=False)
    args = parser.parse_args()

    configure_logging({"LOG_LEVEL": logging.INFO, "LOG_STDOUT": True})

    # By default we run the 'main' function.
    loop = LoopingCall(main, args.upload)

    lcdeferred = loop.start(5)
    lcdeferred.addErrback(lambda failure: loop_failed(failure, loop))

    sys.exit(reactor.run())
