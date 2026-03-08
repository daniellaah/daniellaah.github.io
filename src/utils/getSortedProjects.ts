import type { CollectionEntry } from "astro:content";

const statusRank = {
  shipping: 0,
  active: 1,
  lab: 2,
  archived: 3,
} as const;

export default function getSortedProjects(
  projects: CollectionEntry<"projects">[]
) {
  return projects
    .filter(({ data }) => !data.draft)
    .sort((a, b) => {
      if (a.data.featured !== b.data.featured) {
        return Number(b.data.featured) - Number(a.data.featured);
      }

      if (a.data.order !== b.data.order) {
        return a.data.order - b.data.order;
      }

      if (a.data.year !== b.data.year) {
        return (b.data.year ?? 0) - (a.data.year ?? 0);
      }

      return statusRank[a.data.status] - statusRank[b.data.status];
    });
}
