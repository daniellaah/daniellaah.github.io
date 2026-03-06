import { defineCollection, z } from "astro:content";
import { SITE } from "@/config";

export const BLOG_PATH = "src/content/blog";

const blog = defineCollection({
  schema: ({ image }) =>
    z.object({
      author: z.string().default(SITE.author),
      pubDatetime: z.coerce.date(),
      modDatetime: z.coerce.date().optional().nullable(),
      title: z.string(),
      featured: z.boolean().optional(),
      draft: z.boolean().optional(),
      tags: z.array(z.string()).default(["others"]),
      lang: z.enum(SITE.supportedLangs),
      ogImage: image().or(z.string()).optional(),
      description: z.string(),
      canonicalURL: z.string().optional(),
      timezone: z.string().optional(),
    }),
});

export const collections = { blog };
