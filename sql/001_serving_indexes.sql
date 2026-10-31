-- Manual setup only. Review against your schema/indexes; this file is never run by the app.
-- Run as the database owner in a maintenance window: these normal index builds can block writes.
-- Indexes support the active history, product, profile, and popularity queries.
begin;

-- A unique, non-null review_id within each shopper is required for keyset history paging.
-- Existing data_transform.py produces unique text review IDs. Abort rather than skip rows.
do $check$
begin
    if exists (select 1 from public."Review" where user_id is not null and review_id is null) then
        raise exception 'Fix null review_id values before enabling keyset history paging';
    end if;
end;
$check$;

create unique index if not exists review_user_history_idx
    on public."Review" (user_id, review_id)
    include (parent_asin, review_rating, recency_weight);

create index if not exists review_product_page_idx
    on public."Review" (parent_asin, review_date desc nulls last, review_id asc);

create index if not exists item_popular_order_idx
    on public."Item" (
        num_reviews desc nulls last, avg_rating desc nulls last, parent_asin asc nulls last
    );

-- Omit these if equivalent primary-key/unique indexes already exist under other names.
create unique index if not exists item_parent_asin_lookup_idx on public."Item" (parent_asin);
create unique index if not exists user_id_lookup_idx on public."User" (user_id);

commit;
