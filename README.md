# Introduction

There are three content types of hexo pages, `post`, `page` and `blog_item`.

To create a `blog_item`, type:

```bash
hexo new blog_item "blog_item_name"
```

Then new content type `blog_item` is registered in the file `scripts/blog_item.js`, the layout for the blog list page is defined in `themes/Academia/layout/blog_list.pug`. The css file controlling the style of the blog page is `themes/Academia/source/css/blog.styl`.

There are two classes defined in the blog css file, `blog_list_page` and `blog_page`. To modify the style of the blog page, please refer to these classes.