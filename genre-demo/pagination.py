'''

	Create a pagination navbar for search results.

'''
import math
from flask import render_template
from urllib.parse import urlencode

class Pagination:
	def __init__(self, query_str, query_len, page_num, page_len, max_nav = 7, other_arguments = {}):
		self.query_length = query_len
		self.page_number = page_num
		self.page_length = page_len
		self.query = query_str
		self.total_pages = math.ceil(self.query_length / self.page_length)
		self.max_navigation_links = max_nav # max number of navigation links.
		self.other_arguments = '' if len(other_arguments) == 0 else '&' + urlencode(other_arguments)

		self.min_page = max(self.page_number - math.floor(self.max_navigation_links / 2), 1)
		self.max_page = min(self.page_number + math.floor(self.max_navigation_links / 2), self.total_pages)
		self.trailing_pages = self.page_number - math.floor(self.max_navigation_links / 2) > 1
		self.leading_pages = self.page_number + math.floor(self.max_navigation_links / 2) < self.total_pages

		if self.page_number < math.floor(self.max_navigation_links / 2):
			self.max_page = min(math.floor(self.max_navigation_links), self.total_pages)

		if self.page_number > self.total_pages - math.floor(self.max_navigation_links / 2):
			self.min_page = max(-math.floor(self.max_navigation_links) + self.total_pages + 1, 1)

	def render_navigation(self):
		return render_template('nav-pagination.html', 
								max_navigation_links = self.max_navigation_links,
								query = self.query,
								page_number = self.page_number,
								page_length = self.page_length,
								query_length = self.query_length,
								total_pages = self.total_pages,
								min_page = self.min_page,
								max_page = self.max_page,
								leading_pages = self.leading_pages,
								trailing_pages = self.trailing_pages,
								other_arguments = self.other_arguments)