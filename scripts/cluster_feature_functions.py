import pandas as pd
import numpy as np

def find_allfeatures(FY, client):
    """
    Returns:
        All features for growth model
    """
    print('Finding EVERYTHING')
    query_text = f"""
    DECLARE FY INT64;
    SET FY = {FY};
    
    WITH
      age_articles AS ( # Getting article ages as number of years
        SELECT
          merged_id,
          cluster_id,
          year,
          FY - year AS age,
          year = FY AS is_1_yr,
          year IN (FY-1, FY-2, FY-3, FY-4) AS is_1_4_yr,
        FROM map_of_science.cluster_assignment
        INNER JOIN literature.papers USING(merged_id)
        WHERE (year >= FY - 15) AND (year <= FY)
      ),
    
      #########
      # Finding L0, the stage from peak year
      #########
    
      year_counts_clusters AS ( #Yearly article counts for clusters
        SELECT
          cluster_id,
          year,
          COUNT(DISTINCT merged_id) AS n_articles
        FROM age_articles
        GROUP BY cluster_id, year
      ),
      year_counts_all AS ( #Yearly article counts for full corpus
        SELECT
          year,
          COUNT(merged_id) AS N_G_articles
        FROM age_articles
        GROUP BY year
      ),
      global_share AS ( # Yearly article fraction for each cluster
        SELECT
          cluster_id,
          year,
          COALESCE(N_articles,0)/N_G_articles AS N_share,
          COALESCE(N_articles,0) AS N_articles,
          N_G_articles
        FROM year_counts_all
        LEFT JOIN year_counts_clusters USING(year)
      ),
      cluster_article_count_ranks AS ( # Ranking article share counts for each cluster
        SELECT
          cluster_id,
          year AS peak_year,
          N_share,
          ROW_NUMBER() OVER(PARTITION BY cluster_id ORDER BY N_share DESC, year DESC) AS cl_year_rank
        FROM global_share
      ),
      cluster_peak_years AS ( # Getting peak year for each cluster
        SELECT
          cluster_id,
          peak_year,
          1/(FY - peak_year + 1) AS L0
        FROM cluster_article_count_ranks
        WHERE cl_year_rank = 1
      ),
    
      #########
      # Finding L1, the cluster vitality
      #########
    
      paper_vitality AS (
        SELECT
          cluster_id,
          AVG(1/(age+1)) AS L1
        FROM age_articles
        GROUP BY cluster_id
      ),
    
      #########
      # Finding L2, L3, the reference vitality and change in reference vitality
      #########
    
      age_references AS ( # Age of references
        SELECT
          references.merged_id,
          mergedage.cluster_id AS cluster_id_merged_id,
          ref_id,
          refage.age AS age_ref_id,
          mergedage.year AS year_merged_id
        FROM literature.references
        INNER JOIN age_articles AS mergedage USING(merged_id)
        INNER JOIN age_articles AS refage ON references.ref_id = refage.merged_id
      ),
      age_paper_refs_all AS ( # Average age of paper references
        SELECT
          merged_id,
          cluster_id_merged_id,
          year_merged_id,
          AVG(age_ref_id) AS pap_ref_age_avg
        FROM age_references
        GROUP BY merged_id, year_merged_id, cluster_id_merged_id
      ),
      ref_vitality_all AS ( # Average cluster vitality across all papers
        SELECT
          cluster_id_merged_id AS cluster_id,
          AVG(1/(pap_ref_age_avg + 1)) AS rvit_all
        FROM age_paper_refs_all
        GROUP BY cluster_id_merged_id
      ),
      ref_vitality_FY AS ( # Average cluster vitality for FY
        SELECT
          cluster_id_merged_id AS cluster_id,
          AVG(1/(pap_ref_age_avg + 1)) AS rvit_FY
        FROM age_paper_refs_all
        WHERE year_merged_id = FY
        GROUP BY cluster_id_merged_id
      ),
    
      #########
      # Finding L4, citation vitality
      #########
    
      citation_ages AS ( # Age of citations
        SELECT
          references.merged_id AS cit_id,
          ref_id AS merged_id,
          a1.age AS age_cit,
          a1.year AS year_cit,
          a1.cluster_id AS cluster_id_cit_id,
          a2.age AS age_merged_id,
          a2.year AS year_merged_id,
          a2.cluster_id AS cluster_id_merged_id
        FROM literature.references
        INNER JOIN age_articles AS a1 USING(merged_id)
        INNER JOIN age_articles AS a2 ON references.ref_id = a2.merged_id
      ),
      paper_cit_ages AS ( # Paper avg. age of citations
        SELECT
          merged_id,
          AVG(age_cit) AS paper_avg_cit_age
        FROM citation_ages
        GROUP BY merged_id
      ),
      cluster_cit_ages AS ( # Finding average cluster citation vitality
        SELECT
          cluster_id,
          AVG(1/(paper_avg_cit_age + 1)) AS L4
        FROM age_articles
        INNER JOIN paper_cit_ages USING(merged_id)
        GROUP BY cluster_id
      ),
    
      #########
      # Finding L5, Relative reference age
      #########
    
      paper_ref_diff_ages AS ( # Finding relative age difference between articles and their references
        SELECT
        DISTINCT
          references.merged_id,
          ref_id,
          a1.year - a2.year AS year_diff
        FROM literature.references
        INNER JOIN age_articles AS a1 USING(merged_id)
        INNER JOIN age_articles AS a2 ON references.ref_id = a2.merged_id
      ),
      relative_ref_age_diff AS ( # Finding average relative age difference between areticles and their references
        SELECT
          merged_id,
          AVG(IF(year_diff < 0, 0, year_diff)) AS paper_ref_age
        FROM paper_ref_diff_ages
        GROUP BY merged_id
      ),
      cluster_ref_age_diff AS ( # Finding average relative age difference for papers in a cluster
        SELECT
          cluster_id,
        AVG(paper_ref_age) AS L5
        FROM age_articles
        INNER JOIN relative_ref_age_diff USING(merged_id)
        GROUP BY cluster_id
      ),
    
      #########
      # Finding L6, Relative reference age
      #########
    
      ref_last2yrs AS ( # Fraction of references from last 2 years
        SELECT
          merged_id,
          COUNT(IF(year_diff <= 2, ref_id, NULL)) / COUNT(ref_id) AS frac_last2yrs
        FROM paper_ref_diff_ages
        GROUP BY merged_id
      ),
      cluster_ref_last2yrs AS ( #Average fraction of papers from last 2 years per cluster
        SELECT
          cluster_id,
          AVG(frac_last2yrs) AS L6
        FROM age_articles
        INNER JOIN ref_last2yrs USING(merged_id)
        GROUP BY cluster_id
      ),
    
      #########
      # Finding L7, L8, Growth rates from last 2 years
      #########
    
      yearly_cluster_counts_pivot AS ( # Counting articles in each cluster from last 4 years
        SELECT
          *
        FROM (SELECT merged_id, cluster_id, year FROM age_articles)
        PIVOT(COUNT(DISTINCT merged_id) FOR year IN (FY AS nFY, FY-1 AS nFYm1, FY-2 AS nFYm2, FY-3 AS nFYm3, FY-4 AS nFYm4, FY-5 AS nFYm5, FY-6 AS nFYm6, FY-7 AS nFYm7, FY-8 AS nFYm8, FY-9 AS nFYm9, FY-10 AS nFYm10, FY-11 AS nFYm11, FY-12 AS nFYm12, FY-13 AS nFYm13, FY-14 AS nFYm14, FY-15 AS nFYm15))
      ),
      growth_numerator_denominator_rate2yrs AS ( # Getting numerators and denominators for yearly fractions
        SELECT
          cluster_id,
          nFY AS n1,
          nFYm2 AS d1,
          (nFY + nFYm1 + nFYm2)/3 AS n2,
          (nFYm2 + nFYm3 + nFYm4)/3 AS d2,
        FROM yearly_cluster_counts_pivot
      ),
      growth_rate2yrs AS ( # Growth rate and smoothed growth rate for last 2 years
        SELECT
          cluster_id,
          IF(d1 > 0, n1/d1, 0) AS L7,
          IF(d2 > 0, n2/d2, 0) AS L8
        FROM growth_numerator_denominator_rate2yrs
      ),
    
      #########
      # Finding L9, L10, Recent cluster citations
      #########
    
      new_cluster_cits AS ( #Finding number of recent citations, recent within-cluster citations, and papers
        SELECT
          cluster_id_merged_id AS cluster_id,
          COUNT(cit_id) AS n_recent_citations,
          COUNT(IF(cluster_id_merged_id = cluster_id_cit_id, cit_id, NULL)) AS n_recent_citations_within,
          COUNT(merged_id) AS n_papers
        FROM citation_ages
        WHERE year_cit >= FY - 2
        GROUP BY cluster_id_merged_id
      ),
    
      #########
      # Finding L11 to L15, "Dynamism" (Avila 2013)
      #########
    
      cumulative_paper_counts AS ( # Finding cumulative number of papers for last 4 years
        SELECT
          cluster_id,
          SUM(IF(year <= FY, n_articles, 0)) AS nFY,
          SUM(IF(year <= FY-1, n_articles, 0)) AS nFY1,
          SUM(IF(year <= FY-2, n_articles, 0)) AS nFY2,
          SUM(IF(year <= FY-3, n_articles, 0)) AS nFY3,
        FROM year_counts_clusters
        GROUP BY cluster_id
      ),
      cumulative_fracs AS ( # Finding cumulative number of papers for each cluster as a fraction
        SELECT
          cluster_id,
          nFY/(SELECT SUM(nFY) FROM cumulative_paper_counts) AS frac_FY,
          nFY1/(SELECT SUM(nFY1) FROM cumulative_paper_counts) AS frac_FY1,
          nFY2/(SELECT SUM(nFY2) FROM cumulative_paper_counts) AS frac_FY2,
          nFY3/(SELECT SUM(nFY3) FROM cumulative_paper_counts) AS frac_FY3
        FROM cumulative_paper_counts
      ),
      cumulative_fracs_slopes AS ( # Slopes of change in cumulative fraction of papers
        SELECT
          cluster_id,
          (frac_FY - frac_FY3)/3 AS L11,
          ((frac_FY-frac_FY1) - (frac_FY2 - frac_FY3))/3 AS L14
        FROM cumulative_fracs
      ),
      year_cit_counts AS ( # Yearly citation counts for clusters
        SELECT
          cluster_id_merged_id AS cluster_id,
          year_merged_id AS year,
          COUNT(cit_id) AS n_cits
        FROM citation_ages
        GROUP BY cluster_id_merged_id, year_merged_id
      ),
      cumulative_cits AS ( # Accumulated citation counts
        SELECT
          cluster_id,
          SUM(IF(year <= FY, n_cits, 0)) AS nFY,
          SUM(IF(year <= FY-1, n_cits, 0)) AS nFY1,
          SUM(IF(year <= FY-2, n_cits, 0)) AS nFY2,
          SUM(IF(year <= FY-3, n_cits, 0)) AS nFY3,
        FROM year_cit_counts
        GROUP BY cluster_id
      ),
      cumulative_fracs_cits AS ( # Cumulated citation fractions
        SELECT
          cluster_id,
          nFY/(SELECT SUM(nFY) FROM cumulative_cits) AS frac_FY,
          nFY1/(SELECT SUM(nFY1) FROM cumulative_cits) AS frac_FY1,
          nFY2/(SELECT SUM(nFY2) FROM cumulative_cits) AS frac_FY2,
          nFY3/(SELECT SUM(nFY3) FROM cumulative_cits) AS frac_FY3
        FROM cumulative_cits
      ),
      cumulative_fracs_cits_slopes AS ( # Slopes of change in cumulative fraction of citations
        SELECT
          cluster_id,
          (frac_FY - frac_FY3)/3 AS L12,
          ((frac_FY-frac_FY1) - (frac_FY2 - frac_FY3))/3 AS L15
        FROM cumulative_fracs_cits
      ),
    
      #########
      # Finding S0-S15, dS0 - dS14, Share of PAPERS in each cluster and change in shares of PAPERS in each cluster
      # Finding CS0-CS15, dCS0 - dCS14, Share of CITATIONS in each cluster and change in shares of CITATIONS in each cluster
      # Finding RS0-RS15, dRS0 - dRS14, Share of REFERENCES in each cluster and change in shares of REFERENCES in each cluster
      #########
    
      cluster_shares AS ( # Yearly count of cluster papers as a fraction of each year
        SELECT
          cluster_id,
          nFY/(SELECT N_G_articles FROM year_counts_all WHERE year = FY) AS S0,
          nFYm1/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-1) AS S1,
          nFYm2/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-2) AS S2,
          nFYm3/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-3) AS S3,
          nFYm4/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-4) AS S4,
          nFYm5/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-5) AS S5,
          nFYm6/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-6) AS S6,
          nFYm7/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-7) AS S7,
          nFYm8/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-8) AS S8,
          nFYm9/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-9) AS S9,
          nFYm10/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-10) AS S10,
          nFYm11/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-11) AS S11,
          nFYm12/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-12) AS S12,
          nFYm13/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-13) AS S13,
          nFYm14/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-14) AS S14,
          nFYm15/(SELECT N_G_articles FROM year_counts_all WHERE year = FY-15) AS S15
        FROM yearly_cluster_counts_pivot
      ),
      yearly_cluster_citations_pivot AS ( # Counting citations to each cluster
        SELECT
          *
        FROM (SELECT cit_id, cluster_id_merged_id AS cluster_id, year_cit AS year FROM citation_ages)
        PIVOT(COUNT(DISTINCT cit_id) FOR year IN (FY AS nFY, FY-1 AS nFYm1, FY-2 AS nFYm2, FY-3 AS nFYm3, FY-4 AS nFYm4, FY-5 AS nFYm5, FY-6 AS nFYm6, FY-7 AS nFYm7, FY-8 AS nFYm8, FY-9 AS nFYm9, FY-10 AS nFYm10, FY-11 AS nFYm11, FY-12 AS nFYm12, FY-13 AS nFYm13, FY-14 AS nFYm14, FY-15 AS nFYm15))
      ),
      yearly_cluster_references_pivot AS ( # Counting references from each cluster
        SELECT
          *
        FROM (SELECT merged_id, cluster_id_merged_id AS cluster_id, year_merged_id AS year FROM citation_ages)
        PIVOT(COUNT(DISTINCT merged_id) FOR year IN (FY AS nFY, FY-1 AS nFYm1, FY-2 AS nFYm2, FY-3 AS nFYm3, FY-4 AS nFYm4, FY-5 AS nFYm5, FY-6 AS nFYm6, FY-7 AS nFYm7, FY-8 AS nFYm8, FY-9 AS nFYm9, FY-10 AS nFYm10, FY-11 AS nFYm11, FY-12 AS nFYm12, FY-13 AS nFYm13, FY-14 AS nFYm14, FY-15 AS nFYm15))
      ),
      cluster_citations_share AS (  # Yearly count of cluster citations as a fraction of each year
      SELECT
        cluster_id,
          nFY/(SELECT SUM(nFY) FROM yearly_cluster_citations_pivot) AS CS0,
          nFYm1/(SELECT SUM(nFYm1) FROM yearly_cluster_citations_pivot) AS CS1,
          nFYm2/(SELECT SUM(nFYm2) FROM yearly_cluster_citations_pivot) AS CS2,
          nFYm3/(SELECT SUM(nFYm3) FROM yearly_cluster_citations_pivot) AS CS3,
          nFYm4/(SELECT SUM(nFYm4) FROM yearly_cluster_citations_pivot) AS CS4,
          nFYm5/(SELECT SUM(nFYm5) FROM yearly_cluster_citations_pivot) AS CS5,
          nFYm6/(SELECT SUM(nFYm6) FROM yearly_cluster_citations_pivot) AS CS6,
          nFYm7/(SELECT SUM(nFYm7) FROM yearly_cluster_citations_pivot) AS CS7,
          nFYm8/(SELECT SUM(nFYm8) FROM yearly_cluster_citations_pivot) AS CS8,
          nFYm9/(SELECT SUM(nFYm9) FROM yearly_cluster_citations_pivot) AS CS9,
          nFYm10/(SELECT SUM(nFYm10) FROM yearly_cluster_citations_pivot) AS CS10,
          nFYm11/(SELECT SUM(nFYm11) FROM yearly_cluster_citations_pivot) AS CS11,
          nFYm12/(SELECT SUM(nFYm12) FROM yearly_cluster_citations_pivot) AS CS12,
          nFYm13/(SELECT SUM(nFYm13) FROM yearly_cluster_citations_pivot) AS CS13,
          nFYm14/(SELECT SUM(nFYm14) FROM yearly_cluster_citations_pivot) AS CS14,
          nFYm15/(SELECT SUM(nFYm15) FROM yearly_cluster_citations_pivot) AS CS15
      FROM yearly_cluster_citations_pivot
      ),
      cluster_references_share AS (  # Yearly count of cluster references as a fraction of each year
      SELECT
        cluster_id,
          nFY/(SELECT SUM(nFY) FROM yearly_cluster_references_pivot) AS RS0,
          nFYm1/(SELECT SUM(nFYm1) FROM yearly_cluster_references_pivot) AS RS1,
          nFYm2/(SELECT SUM(nFYm2) FROM yearly_cluster_references_pivot) AS RS2,
          nFYm3/(SELECT SUM(nFYm3) FROM yearly_cluster_references_pivot) AS RS3,
          nFYm4/(SELECT SUM(nFYm4) FROM yearly_cluster_references_pivot) AS RS4,
          nFYm5/(SELECT SUM(nFYm5) FROM yearly_cluster_references_pivot) AS RS5,
          nFYm6/(SELECT SUM(nFYm6) FROM yearly_cluster_references_pivot) AS RS6,
          nFYm7/(SELECT SUM(nFYm7) FROM yearly_cluster_references_pivot) AS RS7,
          nFYm8/(SELECT SUM(nFYm8) FROM yearly_cluster_references_pivot) AS RS8,
          nFYm9/(SELECT SUM(nFYm9) FROM yearly_cluster_references_pivot) AS RS9,
          nFYm10/(SELECT SUM(nFYm10) FROM yearly_cluster_references_pivot) AS RS10,
          nFYm11/(SELECT SUM(nFYm11) FROM yearly_cluster_references_pivot) AS RS11,
          nFYm12/(SELECT SUM(nFYm12) FROM yearly_cluster_references_pivot) AS RS12,
          nFYm13/(SELECT SUM(nFYm13) FROM yearly_cluster_references_pivot) AS RS13,
          nFYm14/(SELECT SUM(nFYm14) FROM yearly_cluster_references_pivot) AS RS14,
          nFYm15/(SELECT SUM(nFYm15) FROM yearly_cluster_references_pivot) AS RS15
      FROM yearly_cluster_references_pivot
      ),
    
      #########
      # Finding I0, I1
      #########
    
      article_venues AS ( #Publication venues
        SELECT
          merged_id,
          cluster_id,
          issn,
          source_name,
        FROM age_articles
        INNER JOIN literature.venues USING(merged_id)
      ),
      articles_last_yr AS ( #Publications from the last year
        SELECT
          merged_id,
          cluster_id
        FROM age_articles
        WHERE is_1_yr
      ),
      articles_1_4_yrs AS ( #Publications with locations from last 1-4 years
        SELECT
          merged_id,
          cluster_id,
          issn,
          source_name
        FROM age_articles
        INNER JOIN article_venues USING(merged_id,cluster_id)
        WHERE is_1_4_yr
      ),
      venue_counts AS ( # Counting articles in each veneu from last 1-4 years
        SELECT
          issn,
          source_name,
          COUNT(DISTINCT merged_id) AS N_articles
        FROM articles_1_4_yrs
        GROUP BY issn, source_name
      ),
      venue_cits AS ( # Getting venue citations for articles
        SELECT
          articles_1_4_yrs.issn,
          articles_1_4_yrs.source_name,
          articles_1_4_yrs.merged_id,
          references.merged_id AS cit_id
        FROM articles_1_4_yrs
        INNER JOIN literature.references ON references.ref_id = articles_1_4_yrs.merged_id
        INNER JOIN articles_last_yr ON references.merged_id = articles_last_yr.merged_id
      ),
      venue_cit_counts AS ( # counting venue citations
        SELECT
          issn,
          source_name,
          COUNT(cit_id) AS N_recent_cits
        FROM venue_cits
        GROUP BY issn, source_name
      ),
      journal_metrics AS ( #Getting journal metrics
        SELECT
          issn,
          source_name,
          N_articles,
          N_recent_cits,
          N_recent_cits/N_articles AS raw_metric
        FROM venue_counts
        INNER JOIN venue_cit_counts USING(issn, source_name)
      ),
      top_550_journals AS ( #top journals, scruffy names
        SELECT
          issn,
          source_name,
          LOWER(source_name) AS l_source_name,
          N_articles,
          N_recent_cits,
          raw_metric
        FROM journal_metrics
        WHERE N_articles > 100
        ORDER BY raw_metric DESC LIMIT 550
      ),
      top_250_journal_names AS ( #simple de-duplication from top 550
        SELECT
          l_source_name,
          MAX(raw_metric) AS metric
        FROM top_550_journals
        GROUP BY l_source_name
        ORDER BY metric DESC
        LIMIT 250
      ),
      top_250_journals AS ( # getting all issns for top journals
        SELECT
          issn,
          source_name,
          l_source_name
        FROM top_550_journals
        INNER JOIN top_250_journal_names USING(l_source_name)
      ),
      papers_top_journals AS ( #number of papers in top journals in FY
        SELECT
          cluster_id,
          COUNT(DISTINCT merged_id) AS I0,
        FROM article_venues
        INNER JOIN top_250_journals USING(issn, source_name)
        INNER JOIN age_articles USING(merged_id, cluster_id)
        WHERE year = FY
        GROUP BY cluster_id
      ),
      refs_to_top_journals AS ( # Get all references to top journals
        SELECT
          DISTINCT
            references.merged_id,
            ref_id,
            issn,
            source_name
        FROM literature.references
        INNER JOIN article_venues ON references.ref_id = article_venues.merged_id
        INNER JOIN top_250_journals USING(issn, source_name)
      ),
      cluster_refs_top_journals AS ( # Count number of references from FY to top journals
        SELECT
          cluster_id,
          COUNT(ref_id) AS I1
        FROM age_articles
        INNER JOIN refs_to_top_journals USING(merged_id)
        WHERE year = FY
        GROUP BY cluster_id
      ),
    
      #########
      # Finding D0, D1, D2
      #########
    
      article_reference_counts AS ( # Counting references
        SELECT
          merged_id,
          cluster_id_merged_id AS cluster_id,
          year_merged_id AS year,
          COUNT(ref_id) AS nrefs
        FROM age_references
        GROUP BY merged_id, cluster_id_merged_id, year_merged_id
      ),
      ref_articles AS ( # Flagging if an article from FY is a review article or not
        SELECT
          merged_id,
          cluster_id,
          nrefs,
          (nrefs > 99) AND (nrefs < 1001) AS is_review
        FROM article_reference_counts
        WHERE year = FY
      ),
      article_type_counts AS ( # Counting review, non-review, and number of references from FY
        SELECT
          cluster_id,
          COUNT(IF(is_review, merged_id, NULL)) AS D0,
          COUNT(IF(is_review, NULL, merged_id)) AS D1,
          SUM(nrefs)  AS D2
        FROM ref_articles
        GROUP BY cluster_id
      ),
    
      #########
      # Finding D3, Number of preprints
      #########
    
      preprint_counts AS ( # Counting number of preprints from the last 2 years from a cluster
        SELECT
          cluster_id,
          COUNT(DISTINCT merged_id) AS D3
        FROM age_articles
        INNER JOIN literature.sources USING(merged_id)
        WHERE (dataset = 'arxiv') AND (year >= FY - 2)
        GROUP BY cluster_id
      ),
    
      #########
      # Finding F0 - F11, Fraction of field-related papers
      #########
    
      article_fields AS ( #labeling articles with fields & mapping to class_art
        SELECT
          merged_id,
          cluster_id,
          top_l0,
          class_art
        FROM age_articles
        INNER JOIN fields_of_study_v2.top_fields USING(merged_id)
        INNER JOIN staging_map_of_science.mapped_fields ON top_l0 = name
      ),
      cluster_field_counts AS ( # field_counts
        SELECT
          *
        FROM (SELECT cluster_id, class_art, merged_id FROM article_fields)
        PIVOT(COUNT(DISTINCT merged_id)
        FOR class_art IN ('biology','business','chemistry','computer science','earth science','engineering','humanities','materials science','mathematics','medicine','physics','social science')
        )
    
      ),
      cluster_sizes AS ( #cluster sizes
        SELECT
          cluster_id,
          COUNT(DISTINCT merged_id) AS n_articles
        FROM article_fields
        GROUP BY cluster_id
      ),
      cluster_field_fracs AS ( #turning field counts into fractions
        SELECT
          cluster_id,
          biology / n_articles AS F0,
          business / n_articles AS F1,
          chemistry / n_articles AS F2,
          `computer science` / n_articles AS F3,
          `earth science` / n_articles AS F4,
          engineering / n_articles AS F5,
          humanities / n_articles AS F6,
          `materials science` / n_articles AS F7,
          mathematics / n_articles AS F8,
          medicine / n_articles AS F9,
          physics / n_articles AS F10,
          `social science` / n_articles AS F11
        FROM cluster_sizes
        LEFT JOIN cluster_field_counts USING(cluster_id)
      ),
    
      #########
      # Finding N0 - N6, Basic network features
      #########
    
      all_edges AS ( #getting all hybrid weights
        SELECT
          a1.merged_id,
          a1.cluster_id AS cluster_id_merged_id,
          a1.year AS year_merged_id,
          ref_id,
          a2.cluster_id AS cluster_id_ref_id,
          a2.year AS year_ref_id,
          weight
        FROM map_of_science.hybrid_weights
        INNER JOIN age_articles AS a1 ON hybrid_weights.merged_id = a1.merged_id
        INNER JOIN age_articles AS a2 ON hybrid_weights.ref_id = a2.merged_id
      ),
      within_cluster_links AS ( # counting number of within-cluster links
        SELECT
          cluster_id_merged_id AS cluster_id,
          COUNT(IF(year_merged_id = FY, ref_id, NULL)) AS n_within_links_FY,
          COUNT(ref_id) AS n_within_links,
          AVG(weight) AS avg_weight,
        FROM all_edges
        WHERE (cluster_id_merged_id = cluster_id_ref_id)
        GROUP BY cluster_id_merged_id
      ),
      cluster_refs AS ( # finding outgoing links
        SELECT
          cluster_id_merged_id AS cluster_id,
          COUNT(ref_id) AS n_refs,
          COUNT(IF(year_merged_id = FY, ref_id, NULL)) AS n_refs_FY,
          COUNT(IF(cluster_id_merged_id = cluster_id_ref_id, ref_id, NULL)) AS n_edges_outgoing,
          SUM(IF(cluster_id_merged_id = cluster_id_ref_id, weight, 0))  AS within_weight_refs
        FROM all_edges
        GROUP BY cluster_id_merged_id
      ),
      cluster_cits AS ( # finding incomeing links
        SELECT
          cluster_id_ref_id AS cluster_id,
          COUNT(merged_id) AS n_cits,
          COUNT(IF(year_merged_id = FY, merged_id, NULL)) AS n_cits_FY,
          COUNT(IF(cluster_id_merged_id = cluster_id_ref_id, merged_id, NULL)) AS n_edges_incoming,
          SUM(IF(cluster_id_merged_id = cluster_id_ref_id, weight, 0))  AS within_weight_cits
        FROM all_edges
        GROUP BY cluster_id_ref_id
      ),
      cluster_links AS ( # counting all links to/from a cluster
        SELECT
          cluster_id,
          COALESCE(n_refs,0) + COALESCE(n_cits,0) AS n_links,
          COALESCE(n_refs_FY,0) + COALESCE(n_cits_FY,0) AS n_links_FY,
          COALESCE(within_weight_cits, 0) + COALESCE(within_weight_refs, 0) AS within_total_weight,
          COALESCE(n_edges_incoming, 0) + COALESCE(n_edges_outgoing, 0) AS within_total_edges
        FROM cluster_refs
        FULL JOIN cluster_cits USING(cluster_id)
      ),
      cluster_node_counts AS ( # counting cluter nodes
        SELECT
          cluster_id,
          COUNT(DISTINCT merged_id) AS n_articles
        FROM age_articles
        GROUP BY cluster_id
      ),
      network_features AS ( # finding N0 - N6
        SELECT
          cluster_id,
          IF(n_links_FY > 0, n_within_links_FY/n_links_FY, 0) AS N0,
          IF(n_links > 0, n_within_links/n_links, 0) AS N1,
          n_articles AS N2,
          n_within_links AS N3,
          avg_weight AS N4,
          within_total_edges/n_articles AS N5,
          within_total_weight/n_articles AS N6
        FROM cluster_links
        FULL JOIN within_cluster_links USING(cluster_id)
        FULL JOIN cluster_node_counts USING(cluster_id)
      ),
    
      #########
      # Finding P0, Patent citations
      #########
    
      patent_refs AS (
        SELECT
          patent_id,
          COALESCE(publication_references.family_id, dates.family_id, 'X-'||patent_id) AS family_id,
          merged_id,
          cluster_id
        FROM age_articles
        INNER JOIN unified_patents.publication_references USING(merged_id)
        INNER JOIN unified_patents.dates USING(patent_id)
        WHERE EXTRACT(YEAR FROM publication_date) <= FY
      ),
      patent_connections AS (
        SELECT
          cluster_id,
          COUNT(DISTINCT family_id||merged_id) AS P0
        FROM patent_refs
        GROUP BY cluster_id
      ),
    
      #########
      # Finding C0, Research front CPT
      #########
    
      cluster_cit_age_ranges AS ( # getting total citations and max/min year for citations from cluster
        SELECT
          cluster_id_merged_id AS cluster_id,
          COUNT(cit_id) AS n_cits,
          MAX(year_cit) AS max_year,
          MIN(year_cit) AS min_year
        FROM citation_ages
        GROUP BY cluster_id_merged_id
      ),
      cpt_table AS (
        SELECT
          cluster_id,
          CASE
            WHEN min_year < max_year THEN n_cits * n_articles / (max_year - min_year)
            ELSE n_cits * n_articles
          END AS C0
        FROM cluster_cit_age_ranges
        INNER JOIN cluster_node_counts USING(cluster_id)
      ),
    
      #########
      # Finding C1, C2, "Entropy"
      #########
    
      field_paper_counts AS ( # field counts for each cluster
        SELECT
          cluster_id,
          class_art AS field,
          COUNT(merged_id) AS n_articles_field
        FROM article_fields
        GROUP BY cluster_id, class_art
      ),
      cluster_total_field_counts AS ( # total paper counts for article with a field
        SELECT
          cluster_id,
          COUNT(merged_id) AS n_articles
        FROM article_fields
        GROUP BY cluster_id
      ),
      field_citation_counts AS ( # field citation counts for each cluster
        SELECT
          cluster_id_merged_id AS cluster_id,
          class_art AS field,
          COUNT(cit_id) AS n_cits_field
        FROM citation_ages
        INNER JOIN article_fields ON citation_ages.cit_id = article_fields.merged_id
        GROUP BY cluster_id_merged_id, class_art
      ),
      cluster_total_cits_field_counts AS ( # total citation counts for each cluster with a field
        SELECT
          cluster_id_merged_id AS cluster_id,
          COUNT(cit_id) AS n_cits
        FROM citation_ages
        INNER JOIN article_fields ON citation_ages.cit_id = article_fields.merged_id
        GROUP BY cluster_id_merged_id
      ),
      field_fracs AS ( # turning field counts into fractions
        SELECT
          cluster_id,
          field,
          n_articles_field/n_articles AS frac_field
        FROM field_paper_counts
        LEFT JOIN cluster_total_field_counts USING(cluster_id)
      ),
      field_citation_fracs AS ( # turning field citation counts into fractions
        SELECT
          cluster_id,
          field,
          n_cits_field/n_cits AS frac_field
        FROM field_citation_counts
        LEFT JOIN cluster_total_cits_field_counts USING(cluster_id)
      ),
      paper_entropy AS ( # finding entropy of fields across papers in a cluster
        SELECT
          cluster_id,
          SUM(frac_field * LOG(frac_field)) AS C1
        FROM field_fracs
        GROUP BY cluster_id
      ),
      citation_entropy AS ( # finding entropy of fields across citations in a cluster
        SELECT
          cluster_id,
          SUM(frac_field * LOG(frac_field)) AS C2
        FROM field_citation_fracs
        GROUP BY cluster_id
      ),
    
      #########
      # Finding CC, Countries and Collaborations
      #########
    
      author_counts AS ( # getting info about author and country info for papers in clusters
        SELECT
          merged_id,
          cluster_id,
          year,
          COUNT(DISTINCT author_name) AS n_authors,
          COUNT(DISTINCT country) AS n_countries,
          ARRAY_AGG(DISTINCT country IGNORE NULLS) AS countries
        FROM age_articles
        INNER JOIN literature.authors USING(merged_id)
        GROUP BY merged_id, cluster_id, year
      ),
      author_cluster_counts AS ( # finding number of articles in a cluster with 3+ authors
        SELECT
          cluster_id,
          COUNT(DISTINCT merged_id) AS n_articles,
          COUNT(DISTINCT IF(n_authors >= 3, merged_id, NULL)) AS n_threeplus_authors
        FROM author_counts
        GROUP BY cluster_id
      ),
      country_cluster_colabs AS ( # finding number of articles in a cluster with 2+ countries
        SELECT
          cluster_id,
          COUNT(DISTINCT IF(n_countries > 1, merged_id, NULL)) AS n_twoplus_countries
        FROM author_counts
        GROUP BY cluster_id
      ),
      country_counts AS ( # Finding country counts for different time periods
        SELECT
          cluster_id,
          country,
          COUNT(DISTINCT IF((year <= FY) AND (year >= FY - 3), merged_id, NULL)) AS n_articles_FY,
          COUNT(DISTINCT IF((year <= FY - 4) AND (year >= FY - 7), merged_id, NULL)) AS n_articles_FY4
        FROM author_counts
        CROSS JOIN UNNEST(countries) AS country
        GROUP BY cluster_id, country
      ),
      active_country_counts AS ( # counting number of active countries for different time periods
        SELECT
          cluster_id,
          COUNT(DISTINCT IF(n_articles_FY4 >= 5, country, NULL)) AS n_countries_FY4,
          COUNT(DISTINCT IF(n_articles_FY >= 5, country, NULL)) AS n_countries_FY
        FROM country_counts
        GROUP BY cluster_id
      ),
      collab_table AS ( # getting country and collab numbers
        SELECT
          cluster_id,
          n_countries_FY - n_countries_FY4 AS CC0,
          n_twoplus_countries/n_articles AS CC1,
          n_threeplus_authors/n_articles AS CC2,
        FROM author_cluster_counts
        FULL JOIN country_cluster_colabs USING(cluster_id)
        FULL JOIN active_country_counts USING(cluster_id)
      )
    
    SELECT
      cluster_id,
      FY AS forecast_year,
      COALESCE(L0, 0) AS L0,
      COALESCE(L1, 0) AS L1,
      COALESCE(rvit_FY, 0) AS L2,
      COALESCE(rvit_all - rvit_FY, 0) AS L3,
      COALESCE(L4, 0) AS L4,
      COALESCE(L5, 0) AS L5,
      COALESCE(L6, 0) AS L6,
      COALESCE(L7, 0) AS L7,
      COALESCE(L8, 0) AS L8,
      COALESCE(n_recent_citations, 0) AS L9,
      COALESCE(n_recent_citations_within/n_papers, 0) AS L10,
      COALESCE(L11, 0) AS L11,
      COALESCE(L12, 0) AS L12,
      COALESCE(L12/L11, 0) AS L13,
      COALESCE(L14, 0) AS L14,
      COALESCE(L15, 0) AS L15,
      COALESCE(I0, 0) AS I0,
      COALESCE(I1, 0) AS I1,
      COALESCE(D0, 0) AS D0,
      COALESCE(D1, 0) AS D1,
      COALESCE(D2, 0) AS D2,
      COALESCE(D3, 0) AS D3,
      COALESCE(F0, 0) AS F0,
      COALESCE(F1, 0) AS F1,
      COALESCE(F2, 0) AS F2,
      COALESCE(F3, 0) AS F3,
      COALESCE(F4, 0) AS F4,
      COALESCE(F5, 0) AS F5,
      COALESCE(F6, 0) AS F6,
      COALESCE(F7, 0) AS F7,
      COALESCE(F8, 0) AS F8,
      COALESCE(F9, 0) AS F9,
      COALESCE(F10, 0) AS F10,
      COALESCE(F11, 0) AS F11,
      COALESCE(N0, 0) AS N0,
      COALESCE(N1, 0) AS N1,
      COALESCE(N2, 0) AS N2,
      COALESCE(N3, 0) AS N3,
      COALESCE(N4, 0) AS N4,
      COALESCE(N5, 0) AS N5,
      COALESCE(N6, 0) AS N6,
      COALESCE(P0, 0) AS P0,
      COALESCE(C0, 0) AS C0,
      COALESCE(C1, 0) AS C1,
      COALESCE(C2, 0) AS C2,
      COALESCE(CC0, 0) AS CC0,
      COALESCE(CC1, 0) AS CC1,
      COALESCE(CC2, 0) AS CC2,
      COALESCE(S0, 0) AS S0,
      COALESCE(S1, 0) AS S1,
      COALESCE(S2, 0) AS S2,
      COALESCE(S3, 0) AS S3,
      COALESCE(S4, 0) AS S4,
      COALESCE(S5, 0) AS S5,
      COALESCE(S6, 0) AS S6,
      COALESCE(S7, 0) AS S7,
      COALESCE(S8, 0) AS S8,
      COALESCE(S9, 0) AS S9,
      COALESCE(S10, 0) AS S10,
      COALESCE(S11, 0) AS S11,
      COALESCE(S12, 0) AS S12,
      COALESCE(S13, 0) AS S13,
      COALESCE(S14, 0) AS S14,
      COALESCE(S15, 0) AS S15,
      COALESCE(S0 - S1, 0) AS dS0,
      COALESCE(S1 - S2, 0) AS dS1,
      COALESCE(S2 - S3, 0) AS dS2,
      COALESCE(S3 - S4, 0) AS dS3,
      COALESCE(S4 - S5, 0) AS dS4,
      COALESCE(S5 - S6, 0) AS dS5,
      COALESCE(S6 - S7, 0) AS dS6,
      COALESCE(S7 - S8, 0) AS dS7,
      COALESCE(S8 - S9, 0) AS dS8,
      COALESCE(S9 - S10, 0) AS dS9,
      COALESCE(S10 - S11, 0) AS dS10,
      COALESCE(S11 - S12, 0) AS dS11,
      COALESCE(S12 - S13, 0) AS dS12,
      COALESCE(S13 - S14, 0) AS dS13,
      COALESCE(S14 - S15, 0) AS dS14,
      COALESCE(CS0, 0) AS CS0,
      COALESCE(CS1, 0) AS CS1,
      COALESCE(CS2, 0) AS CS2,
      COALESCE(CS3, 0) AS CS3,
      COALESCE(CS4, 0) AS CS4,
      COALESCE(CS5, 0) AS CS5,
      COALESCE(CS6, 0) AS CS6,
      COALESCE(CS7, 0) AS CS7,
      COALESCE(CS8, 0) AS CS8,
      COALESCE(CS9, 0) AS CS9,
      COALESCE(CS10, 0) AS CS10,
      COALESCE(CS11, 0) AS CS11,
      COALESCE(CS12, 0) AS CS12,
      COALESCE(CS13, 0) AS CS13,
      COALESCE(CS14, 0) AS CS14,
      COALESCE(CS15, 0) AS CS15,
      COALESCE(CS0 - CS1, 0) AS dCS0,
      COALESCE(CS1 - CS2, 0) AS dCS1,
      COALESCE(CS2 - CS3, 0) AS dCS2,
      COALESCE(CS3 - CS4, 0) AS dCS3,
      COALESCE(CS4 - CS5, 0) AS dCS4,
      COALESCE(CS5 - CS6, 0) AS dCS5,
      COALESCE(CS6 - CS7, 0) AS dCS6,
      COALESCE(CS7 - CS8, 0) AS dCS7,
      COALESCE(CS8 - CS9, 0) AS dCS8,
      COALESCE(CS9 - CS10, 0) AS dCS9,
      COALESCE(CS10 - CS11, 0) AS dCS10,
      COALESCE(CS11 - CS12, 0) AS dCS11,
      COALESCE(CS12 - CS13, 0) AS dCS12,
      COALESCE(CS13 - CS14, 0) AS dCS13,
      COALESCE(CS14 - CS15, 0) AS dCS14,
      COALESCE(RS0, 0) AS RS0,
      COALESCE(RS1, 0) AS RS1,
      COALESCE(RS2, 0) AS RS2,
      COALESCE(RS3, 0) AS RS3,
      COALESCE(RS4, 0) AS RS4,
      COALESCE(RS5, 0) AS RS5,
      COALESCE(RS6, 0) AS RS6,
      COALESCE(RS7, 0) AS RS7,
      COALESCE(RS8, 0) AS RS8,
      COALESCE(RS9, 0) AS RS9,
      COALESCE(RS10, 0) AS RS10,
      COALESCE(RS11, 0) AS RS11,
      COALESCE(RS12, 0) AS RS12,
      COALESCE(RS13, 0) AS RS13,
      COALESCE(RS14, 0) AS RS14,
      COALESCE(RS15, 0) AS RS15,
      COALESCE(RS0 - RS1, 0) AS dRS0,
      COALESCE(RS1 - RS2, 0) AS dRS1,
      COALESCE(RS2 - RS3, 0) AS dRS2,
      COALESCE(RS3 - RS4, 0) AS dRS3,
      COALESCE(RS4 - RS5, 0) AS dRS4,
      COALESCE(RS5 - RS6, 0) AS dRS5,
      COALESCE(RS6 - RS7, 0) AS dRS6,
      COALESCE(RS7 - RS8, 0) AS dRS7,
      COALESCE(RS8 - RS9, 0) AS dRS8,
      COALESCE(RS9 - RS10, 0) AS dRS9,
      COALESCE(RS10 - RS11, 0) AS dRS10,
      COALESCE(RS11 - RS12, 0) AS dRS11,
      COALESCE(RS12 - RS13, 0) AS dRS12,
      COALESCE(RS13 - RS14, 0) AS dRS13,
      COALESCE(RS14 - RS15, 0) AS dRS14,
    FROM cluster_peak_years
    FULL JOIN paper_vitality USING(cluster_id)
    FULL JOIN ref_vitality_all USING(cluster_id)
    FULL JOIN ref_vitality_FY USING(cluster_id)
    FULL JOIN cluster_cit_ages USING(cluster_id)
    FULL JOIN cluster_ref_age_diff USING(cluster_id)
    FULL JOIN cluster_ref_last2yrs USING(cluster_id)
    FULL JOIN growth_rate2yrs USING(cluster_id)
    FULL JOIN new_cluster_cits USING(cluster_id)
    FULL JOIN cumulative_fracs_cits_slopes USING(cluster_id)
    FULL JOIN cumulative_fracs_slopes USING(cluster_id)
    FULL JOIN papers_top_journals USING(cluster_id)
    FULL JOIN article_type_counts USING(cluster_id)
    FULL JOIN preprint_counts USING(cluster_id)
    FULL JOIN cluster_field_fracs USING(cluster_id)
    FULL JOIN network_features USING(cluster_id)
    FULL JOIN patent_connections USING(cluster_id)
    FULL JOIN cpt_table USING(cluster_id)
    FULL JOIN paper_entropy USING(cluster_id)
    FULL JOIN citation_entropy USING(cluster_id)
    FULL JOIN collab_table USING(cluster_id)
    FULL JOIN cluster_refs_top_journals USING(cluster_id)
    FULL JOIN cluster_shares USING(cluster_id)
    FULL JOIN cluster_citations_share USING(cluster_id)
    FULL JOIN cluster_references_share USING(cluster_id)
    """

    query = client.query(query_text)
    return pd.DataFrame([{k: row[k] for k in row.keys()} for row in query])
