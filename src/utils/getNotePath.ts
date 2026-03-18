import { getResolvedSlug } from "./contentSlug";

export function getNotePath(id: string, explicitSlug?: string) {
  return `/notes/${getResolvedSlug(id, explicitSlug)}`;
}
