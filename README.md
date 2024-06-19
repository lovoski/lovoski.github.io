# Introduction

There are three content types of hexo pages, `post`, `page` and `blog_item`.

To create a `blog_item`, input the following command in your terminal:

```bash
hexo new blog_item "blog_item_name"
```

The new content type `blog_item` is registered in the file `scripts/blog_item.js`, the layout for the blog list page is defined in `themes/Academia/layout/blog_list.pug`. The css file controlling the style of the blog page is `themes/Academia/source/css/blog.styl`.

There are two classes defined in the blog css file, `blog_list_page` and `blog_page`. To modify the style of the blog page, please refer to these classes.

## Configuration

### Math function rendering

As mentioned above, this webpage is built with hexo, but the default renderer `hexo-renderer-markd` doesn't come with support for math function rendering. So I uninstalled the default renderer and switched to `hexo-renderer-pandoc` and `hexo-filter-mathjax` for equation rendering. Relavent configuration can be found in file `/_config.yml`:

```yml
mathjax:
  tags: none # or 'ams' or 'all'
  single_dollars: true # enable single dollar signs as in-line math delimiters
  cjk_width: 0.9 # relative CJK char width
  normal_width: 0.6 # relative normal (monospace) width
  append_css: true # add CSS to pages rendered by MathJax
  every_page: false # if true, every page will be rendered by MathJax regardless the `mathjax` setting in Front-matter
```

To ensure the webpage can be compiled successfully with pandoc, a distribution of the program should be installed and set to PATH environment variable.

Also, to ensure the math function rendered, one tag `mathjax: true` should be included in the front-matter of the markdown page.
