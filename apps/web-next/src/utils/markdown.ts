import MarkdownIt from "markdown-it";
import hljs from "highlight.js/lib/core";
import cpp from "highlight.js/lib/languages/cpp";
import python from "highlight.js/lib/languages/python";
import bash from "highlight.js/lib/languages/bash";
import json from "highlight.js/lib/languages/json";

hljs.registerLanguage("c", cpp);
hljs.registerLanguage("cpp", cpp);
hljs.registerLanguage("python", python);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("json", json);

// markdown-it escapes raw HTML and validates link schemes. Model text is not trusted HTML.
const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  highlight: (code, language) =>
    language && hljs.getLanguage(language)
      ? hljs.highlight(code, { language }).value
      : "",
});
const renderLink =
  markdown.renderer.rules.link_open ??
  ((tokens, index, options, _env, renderer) =>
    renderer.renderToken(tokens, index, options));
markdown.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
  tokens[index]!.attrSet("target", "_blank");
  tokens[index]!.attrSet("rel", "noopener noreferrer");
  return renderLink(tokens, index, options, env, renderer);
};
export const renderMarkdown = (text: string) => markdown.render(text);
