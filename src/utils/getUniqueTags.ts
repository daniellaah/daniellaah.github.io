import { slugifyStr } from "./slugify";

interface Tag {
  tag: string;
  tagName: string;
}

type TaggedEntry = {
  data: {
    tags: string[];
    draft?: boolean;
  };
};

const getUniqueTags = <T extends TaggedEntry>(entries: T[]) => {
  const tags: Tag[] = entries
    .filter(({ data }) => !data.draft)
    .flatMap(entry => entry.data.tags)
    .map(tag => ({ tag: slugifyStr(tag), tagName: tag }))
    .filter(
      (value, index, self) =>
        self.findIndex(tag => tag.tag === value.tag) === index
    )
    .sort((tagA, tagB) => tagA.tag.localeCompare(tagB.tag));
  return tags;
};

export default getUniqueTags;
