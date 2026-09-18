# Product ID audit

The generated [product ID evidence table](product-id-audit.csv) covers all **116 `(website, review_local_id)` pairs** in the 81,377 supplied review rows. It gives review counts, names from two legacy tables, and counts of product style codes mentioned in review text. Every row is marked `UNVERIFIED_REQUIRES_ORIGINAL_PRODUCT_URL_OR_EXPORT`; the table is for manual source tracing, not a forecast join.

The review pickles have **109 numeric IDs**, but seven numbers occur on both websites. A numeric ID alone is therefore not a unique product key. The earlier `data_sentiment_merged.pkl` uses keys such as `Levis_46` and `Macys_115`, which can be joined to review site plus numeric ID for 43 of the 116 pairs (33,113 of 81,377 review rows). That match is suggestive but not proven to represent the same scraping/catalog version. The later v3 table renumbers products into 0–140 (source 1: 0–78; source 2: 79–140), so its numeric `unique_id` cannot be joined to a review's numeric ID.

Two concrete contradictions show why no 505 mapping is approved:

| Review group | Early prefixed table | Later numeric table at the same number | Review text clue |
| --- | --- | --- | --- |
| Levi’s ID 14, 9,253 rows | “511 Slim Fit Men's Jeans” | “311 Shaping Skinny Women's Jeans” | 2,177 reviews mention `505`; 14 mention `511`. Text mentions can include comparisons, so this does not prove the product is 505. |
| Macy’s ID 41, 5,568 rows | No matching `Macys_41` record | “505 Regular Fit Men's Jeans” | 2,830 reviews mention `501`; only 60 mention `505`. The later numeric label is particularly implausible for this review group. |

The earlier table names four Levi’s local IDs as 505 products (46, 50, 51, 52) and two Macy’s IDs (115, 118). Among the supplied reviews, only Levi’s ID 46 (168 rows) and ID 51 (373 rows) have matching site/ID keys. Their text mentions 505 zero and six times respectively, so even these are candidates rather than verified joins. The other four table keys have no supplied review rows. Product strings can change across catalog versions, and review text often compares products, making keyword matching insufficient.

To verify a mapping, recover the original product page URL, source product identifier, or the scrape export that contained both the review and the product ID/name at collection time. Validate `(website, source_product_id)` uniqueness and effective dates, then spot-check examples from each mapped group. Until then, exclude the review pickles from 505-specific forecasting features. The separate weekly Trends target itself explicitly names “505 Levi” in its export header.

Regenerate this table with `python scripts/prepare_added_data.py`. Its `v3_numeric_*_UNSAFE_JOIN` columns are shown only to expose collisions; they are **never** approved labels.
