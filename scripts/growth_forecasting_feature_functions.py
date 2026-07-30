import pandas as pd

def growth_ratings(FY,client):
    query_text = f"""
    WITH
    age_articles AS ( #Getting the age of all articles in clusters, keeping 15 yr cutoff
      SELECT
        merged_id,
        cluster_id,
        {FY} - year AS age,
        year
      FROM map_of_science.cluster_assignment
      INNER JOIN literature.papers USING(merged_id)
      WHERE (year >= ({FY}-15)) AND (year <= {FY+5})
    ),
    year_counts_clusters AS ( #Finding yearly article counts for each cluster
      SELECT
        cluster_id,
        year,
        COUNT(DISTINCT merged_id) AS N_articles
      FROM age_articles
      GROUP BY cluster_id, year
    ),
    year_counts_all AS ( #Getting global article counts for all years
      SELECT
        year,
        COUNT(DISTINCT merged_id) AS N_G_articles
      FROM age_articles
      GROUP BY year
    ),
    global_share AS ( #Finding the global share of articles in a cluster per year
      SELECT
        cluster_id,
        year,
        COALESCE(N_articles,0)/N_G_articles AS N_share,
        COALESCE(N_articles,0) AS N_articles,
        N_G_articles
      FROM year_counts_all 
      LEFT JOIN year_counts_clusters USING(year)
    ),
    cluster_article_count_ranks AS ( #Ranking article share counts for each cluster to find peak year
      SELECT
        cluster_id,
        year AS peak_year,
        N_share,
        ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY N_share DESC, year DESC) AS cl_year_rank
      FROM global_share
      WHERE year <= {FY}
    ),
    gs1 AS (
      SELECT
        *
      FROM global_share WHERE year = {FY+1}
    ),
    gs2 AS (
      SELECT
        *
      FROM global_share WHERE year = {FY+2}
    ),
    gs3 AS (
      SELECT
        *
      FROM global_share WHERE year = {FY+3}
    ),
    gs4 AS (
      SELECT
        *
      FROM global_share WHERE year = {FY+4}
    ),
    gs5 AS (
      SELECT
        *
      FROM global_share WHERE year = {FY+5}
    ),
    growth_ratings AS (
        SELECT
        DISTINCT
          cluster_id,
          peak_year,
          c_rank.N_share AS N_share_peak,
          gs1.N_share AS Share1,
          gs2.N_share AS Share2,
          gs3.N_share AS Share3,
          gs4.N_share AS Share4,
          gs5.N_share AS Share5,
          POWER(gs1.N_share/c_rank.N_share,1/({FY+1}-peak_year)) AS GR1,
          POWER(gs2.N_share/c_rank.N_share,1/({FY+2}-peak_year)) AS GR2,
          POWER(gs3.N_share/c_rank.N_share,1/({FY+3}-peak_year)) AS GR3,
          POWER(gs4.N_share/c_rank.N_share,1/({FY+4}-peak_year)) AS GR4,
          POWER(gs5.N_share/c_rank.N_share,1/({FY+5}-peak_year)) AS GR5,
        FROM cluster_article_count_ranks AS c_rank
        LEFT JOIN gs1 USING(cluster_id)
        LEFT JOIN gs2 USING(cluster_id)
        LEFT JOIN gs3 USING(cluster_id)
        LEFT JOIN gs4 USING(cluster_id)
        LEFT JOIN gs5 USING(cluster_id)
        WHERE cl_year_rank = 1
    )
    
    SELECT
    DISTINCT
        cluster_id,
        {FY} AS forecast_year,
        peak_year,
        Share1,
        Share2,
        Share3,
        Share4,
        Share5,
        GR1,
        GR2,
        GR3,
        GR4,
        GR5,
        IF(GR1 > 1.08, 1, 0) AS EG_1Yr,
        IF((GR1 > 1.08) OR (GR2 > 1.08), 1, 0) AS EG_2Yr,
        IF((GR1 > 1.08) OR (GR2 > 1.08) OR (GR3 > 1.08), 1, 0) AS EG_3Yr,
        IF((GR1 > 1.08) OR (GR2 > 1.08) OR (GR3 > 1.08) OR (GR4 > 1.08), 1, 0) AS EG_4Yr,
        IF((GR1 > 1.08) OR (GR2 > 1.08) OR (GR3 > 1.08) OR (GR4 > 1.08) OR (GR4 > 1.08), 1, 0) AS EG_5Yr,
    FROM growth_ratings
    """

    query = client.query(query_text)
    return pd.DataFrame([{k:row[k] for k in row.keys()} for row in query])