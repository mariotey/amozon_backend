-- Manual setup only. Review against your schema/indexes; this file is never run by the app.
-- Run as the database owner in a maintenance window: these normal index builds can block writes.
-- Existing serving semantics: a shopper is eligible if Review has a non-null user_id.
-- Do NOT substitute LIMIT alone for ORDER BY random(): it is not a random sample.
begin;

create or replace function public.random_shoppers(sample_size integer default 5)
returns table(user_id text)
language plpgsql
volatile
security invoker
set search_path = ''
as $function$
begin
    if sample_size is null or sample_size < 1 or sample_size > 100 then
        raise exception 'sample_size must be between 1 and 100' using errcode = '22023';
    end if;
    return query
    select eligible.user_id
    from (
        select distinct r.user_id::text as user_id
        from public."Review" as r
        where r.user_id is not null
    ) as eligible
    order by pg_catalog.random()
    limit sample_size;
end;
$function$;

-- Invoker rights retain the caller's table grants/RLS. The backend credential must
-- have SELECT on User/Item/Review and execute this RPC. Never expose its secret to browsers.
revoke all on function public.random_shoppers(integer) from public, anon, authenticated;
grant execute on function public.random_shoppers(integer) to service_role;

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
    on public."Item" (num_reviews desc nulls last, avg_rating desc nulls last, parent_asin asc nulls last);

-- Omit these if equivalent primary-key/unique indexes already exist under other names.
create unique index if not exists item_parent_asin_lookup_idx on public."Item" (parent_asin);
create unique index if not exists user_id_lookup_idx on public."User" (user_id);

-- Ask PostgREST to refresh its function signature cache after the transaction commits.
notify pgrst, 'reload schema';
commit;
