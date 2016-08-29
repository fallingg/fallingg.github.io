---
layout: archive
permalink: /tags/
title: "Tags"
author_profile: true
---

{% include base_path %}

{% capture site_tags %}{% for tag in site.tags %}{{ tag | first }}{% unless forloop.last %},{% endunless %}{% endfor %}{% endcapture %}
<!-- site_tags: {{ site_tags }} -->
{% assign tag_words = site_tags | split:',' | sort %}
<!-- tag_words: {{ tag_words }} -->
  <ul class="tag-box inline">
  {% for item in (0..site.tags.size) %}{% unless forloop.last %}
    {% capture this_word %}{{ tag_words[item] | strip_newlines }}{% endcapture %}
    <a href="/search/?tags={{ this_word | cgi_escape }}" class="btn btn--info">{{ this_word }} <span>{{ site.tags[this_word].size }}</span></a>
  {% endunless %}{% endfor %}
  </ul>