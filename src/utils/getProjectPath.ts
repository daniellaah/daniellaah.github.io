import { slugifyStr } from "./slugify";

export function getProjectPath(id: string) {
  const slug = id
    .split("/")
    .filter(Boolean)
    .slice(-1)[0]
    ?.replace(/\.(md|mdx)$/i, "");

  return `/projects/${slugifyStr(slug ?? id)}`;
}
