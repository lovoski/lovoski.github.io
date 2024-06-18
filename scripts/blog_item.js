hexo.extend.generator.register('blog_item', function(locals) {
  return {
    path: 'Blog/index.html',
    data: locals.posts.filter(post => post.is_blog_item),
    layout: ['blog_list']
  };
});