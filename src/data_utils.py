import numpy as np
import pandas as pd
import tqdm
import sqlite3
import pycountry
import src.utils as utils

    
    
class TableCache():
    def __init__(self):
        self.cache = {}
    def __getitem__(self, k):
        return self.cache.__getitem__(k)
    def __setitem__(self, k, v):
        return self.cache.__setitem__(k,v)
    def __contains__(self, k):
        return self.cache.__contains__(k)
    def flush(self):
        self.cache = {}
        
class DataLoader:
    def __init__(self, db_con, filedb_con=None):
        if isinstance(db_con, str):
            db_con = sqlite3.connect(db_con)
        self.db_con = db_con
        self.filedb_con = filedb_con
        self.table_processor = TableProcessor()
        self.TABLE_CACHE = TableCache()

    def process_table(self, table_name, table):
        return self.table_processor(table_name, table)

    def flush_table_cache(self, tables=None):
        if tables is None:
            self.TABLE_CACHE.flush()
        if isinstance(tables, str):
            try:
                del self.TABLE_CACHE.cache[tables]
            except:
                pass
        else:
            for tb in tables:
                try:
                    del self.TABLE_CACHE.cache[tb]
                except:
                    pass

    def get_cache_table(self, table, con, **kwargs):
        if table not in self.TABLE_CACHE:
            tb = self.process_table(table, tb)
            self.TABLE_CACHE[table] = tb
        return self.TABLE_CACHE[table].copy()

    def get_table(self, table, con='db', **kwargs):
        table_getters = {
            'author_inst_year_explode': get_author_inst_year_explode,
            'ins_per_year': get_inst_ranking,
            'author_full': get_author_citation,
            'author_full_pivot': get_author_full_pivot,
            'author_grp': get_author_grp,
            'sub_arx_time': get_sub_arx_time,
            'sub_derive': get_sub_derive,
            'sub_reviews_agg': get_review_agg,
            'sub_reviews': get_review_data,
            'sub_full': get_sub_full,
            'grp_ins_per_year': get_group_inst_per_year,


            
        }
        if table not in self.TABLE_CACHE:
            if table in table_getters:
#                 print('getter')
                tb = table_getters[table](self.get_table, db_con=self.db_con, filedb_con=self.filedb_con, **kwargs)
                tb = self.process_table(table, tb)
                self.TABLE_CACHE[table] = tb
            else:   
#                 print(table)
                if isinstance(con, str):
                    tb = utils.get_table(table, con=self.db_con if con=='db' else self.filedb_con, **kwargs)   
                else:
                    tb = utils.get_table(table, con=con, **kwargs)   
                tb = self.process_table(table, tb)
                self.TABLE_CACHE[table] = tb
        return self.TABLE_CACHE[table].copy()
    
    def __call__(self, *args, **kwargs):
        return self.get_table(*args, **kwargs)
    


class TableProcessor():
    def __init__(self):
        self.processors = {
            'submissions': self.sub_proc,
            'author_gscholar_citation': self.author_gscholar_citation_proc,
#             'iclr_dataset': lambda tb: utils.df_col_totype(tb, ['rating_int','input_len','review_len'], int),
            'csrank_area': self.csrank_area_proc,
            'author_submission': self.author_sub_proc,
            'author_gender': lambda tb: utils.df_col_totype(tb,['perceived_gender',],float),
            'submission_summary': self.sub_summary_proc,
            'submission_sentiment': self.sub_sent_proc,
            'author_full': self.author_full_proc,
    }
        
    def __call__(self, table_name, table):
        if table_name in self.processors:
            table = self.processors[table_name](table)
        return table

    @staticmethod
    def author_full_proc(tb):
        tb = utils.df_col_totype(tb,['author_no','n_author','year'],int)
        return tb
    @staticmethod
    def sub_sent_proc(tb):
        tb = utils.df_col_totype(tb,['mean_sentiment'],float)
        return tb
    
    @staticmethod
    def sub_summary_proc(tb):
        tb = utils.df_col_totype(tb,['n_fig', 'n_ref','n_sec'],float)
        return tb
    @staticmethod
    def csrank_area_proc(tb):
        tb = utils.df_col_totype(tb,['adjustedcount','count'],float)
        tb['year'] = tb['year'].astype(int)
        return tb
    @staticmethod
    def sub_proc(tb):
        tb= tb.rename({'id':'submission_id'},axis=1)
        tb['venue']=tb.apply(lambda s : s['conf_name']+s['conf_year'], axis=1)
        tb['conf_year']=tb.conf_year.astype(float).astype(int)
        tb['year']=tb.conf_year.astype(float).astype(int)

        return tb
    
    @staticmethod
    def author_gscholar_citation_proc(tb):
        tb.year = tb.year.astype(float).astype(int)
        tb.cite=tb.cite.astype(float)
        tb = (tb
            .sort_values(['author_id','year'],ascending=True)
            .reset_index(drop=True)
        )
        tb['cum_cite'] =\
            tb.groupby(['author_id']).cite.transform('cumsum')
        tb['cum_cite'] =\
            tb.groupby(['author_id']).cite.transform('cumsum')
        return tb

    @staticmethod
    def author_sub_proc(tb):
        tb['n_author']=tb.groupby("submission_id").author_id.transform('nunique')
        return tb

def get_sub_full(get_table_handle,**kwargs):
    venues = get_table_handle('venues')
    matched_sub = get_table_handle('submission_arxiv')
    submissions = get_table_handle('submissions')
  
    author_grp = get_table_handle('author_grp')
    author_full = get_table_handle('author_full')

    sub_reviews = get_table_handle('sub_reviews')
    sub_reviews_agg = get_table_handle('sub_reviews_agg')
    sub_derive = get_table_handle('sub_derive')
    
    design_dat = (sub_reviews
     .merge(sub_derive,on='submission_id',how='inner')
     .merge(submissions[['submission_id','venue']], on='submission_id')
     .merge(venues,on='venue',how='inner')
     .merge(author_grp,on='submission_id',how='inner')
     .drop(['id','year_x','year_y', 'arxiv_first_y'],axis=1)
     .rename({
         'arxiv_first_x': 'arxiv_first',
         'sentiment': 'review_sentiment',
         'sub_fluency': 'submission_complexity',
         'n_fig':'submission_n_fig',
         'n_ref':'submission_n_ref',
         'n_sec':'submission_n_sec',
     },axis=1)
    )
    # design_dat.conf_year=design_dat.conf_year.astype(int)
    design_dat=utils.df_col_totype(design_dat,['submission_complexity',
                                               'submission_n_fig',
                                               'submission_n_ref',
                                               'submission_n_sec'],float)
    return design_dat
    
def get_author_grp(get_table_handle,**kwargs):
    author_full = get_table_handle('author_full')
    
    return (author_full
        .assign(any_reported_f=lambda x : x.reported_gender,
                cnt_reported_f=lambda x : x.reported_gender,
                fst_reported_f=lambda x : x.reported_gender,
                any_perceived_f=author_full.perceived_gender<0.3,
                cnt_perceived_f=author_full.perceived_gender<0.3,
                fst_perceived_f=author_full.perceived_gender<0.3,
                demo_no_us=lambda x : x.iso_a3,
                demo_mode=lambda x : x.iso_a3,
                ins_rank_max=lambda x : x.ins_per_year_rank,
                ins_rank_avg=lambda x : x.ins_per_year_rank,
                ins_rank_min=lambda x : x.ins_per_year_rank,
                author_cite_max=lambda x : x.author_year_citation,
                author_cite_avg=lambda x : x.author_year_citation,
                author_cite_min=lambda x : x.author_year_citation,

               )
        .sort_values(['submission_id','author_no'])
        .groupby("submission_id")
        .agg({
            'n_author':'first', 'year':'first',
            'any_reported_f': lambda grp: (grp=='Female').any(),
            'cnt_reported_f': lambda grp: (grp=='Female').sum(),
            'fst_reported_f': lambda grp: (grp=='Female').iloc[0],
            'any_perceived_f': lambda grp: (grp==1).any(),
            'cnt_perceived_f': lambda grp: (grp==1).sum(),
            'fst_perceived_f':lambda grp: (grp==1).iloc[0],
            'demo_no_us': lambda grp: (grp!='USA').all(),
            'demo_mode':pd.Series.mode,
            'ins_rank_max':'max','ins_rank_avg':'mean','ins_rank_min':'min',
            'author_cite_max':'max','author_cite_avg':'mean','author_cite_min':'min',
            'arxiv_first':'first',
            'top_inst':lambda grp: (grp==1).any(),
            'top_author':lambda grp: (grp==1).any(),
            
        })
    ).reset_index()

def get_sub_arx_time(get_table_handle,**kwargs):
    submissions = get_table_handle('submissions')
    venues = get_table_handle('venues')
    matched_sub = get_table_handle('submission_arxiv')
    
    sub_arx_time = (submissions
     .merge(venues)
     [['submission_id','review_ddl']]
     .merge(matched_sub[['submission_id', 'primary_category', 'published_time']],
           how='left')
     .rename({'review_ddl':'sub_date'},axis=1)
    )

    sub_arx_time['arxiv_first'] = sub_arx_time.apply(
        lambda s : False if pd.isna(s['published_time']) else (
            True if s['published_time'] < s['sub_date'] else False
        ), axis=1

    )
    return sub_arx_time
    
def get_sub_derive(get_table_handle,**kwargs):
    submissions = get_table_handle('submissions')
    sub_arx_time = get_table_handle('sub_arx_time')
    sub_sentiment =  get_table_handle('submission_sentiment')
    sub_summary = get_table_handle('submission_summary')
    sub_main_keyword = get_table_handle('submission_main_keyword')

    sub_derive = (submissions[['submission_id','year']]
        .merge(sub_sentiment[['submission_id','mean_sentiment']],how='left')
        .rename({'mean_sentiment':'sub_fluency'},axis=1)
        .merge(sub_summary.rename({'id':'submission_id'},axis=1),how='left')    
        .merge(sub_main_keyword,how='left')
        .merge(sub_arx_time[['submission_id','arxiv_first']])
    )
    return sub_derive

def get_author_citation(get_table_handle,**kwargs):
    """
    table dependencies: author_inst_year_explode
    table returned: author_full
    """
    gs_author_year = get_table_handle('author_gscholar_citation')
    submissions = get_table_handle('submissions')
    author_inst_year_explode = get_table_handle('author_inst_year_explode')
    csrank_area = get_table_handle('csrank_area')
    author_gender = get_table_handle('author_gender')
    author_sub = get_table_handle('author_submission')
    ins_per_year = get_table_handle('ins_per_year')
    sub_arx_time = get_table_handle('sub_arx_time')
    
    author_cite_df = (author_sub
     .merge(submissions[['submission_id','year']], how='outer')

     .merge(author_inst_year_explode, on=['author_id','year'], how='left')

     # institution csranking
     .merge(
         (csrank_area.groupby(['institution_id','year'])
                 .sum().reset_index()
                 .rename({'count':'ins_cite_cnt',
                          'adjustedcount':'ins_adj_cnt'
                         },axis=1)
        ), on=['institution_id','year'], how='left'
     )

     # author per-year citation
     .merge(gs_author_year,on=['author_id','year'], how='left')
     .rename({'cum_cite':'citation_count'},axis=1)

     # tidy up
     .drop([ 'institution_id'],axis=1)
    #  .groupby('submission_id').agg(list).reset_index()
    )

    author_cite_df=( author_cite_df
                .sort_values(['author_id','year'],ascending=True)
                .reset_index(drop=True)
               )

    author_cite_df=author_cite_df.set_index(['author_id'])

    author_cite_df=(
        author_cite_df
        .groupby('author_id')
        .transform(lambda v: v.ffill())
        .reset_index()
    )
    author_cite_df['citation_count']=author_cite_df['citation_count'].astype(float)
    author_cite_df['ins_adj_cnt']=author_cite_df['ins_adj_cnt'].astype(float)
    
    author_cite_per_year = (author_cite_df[['author_id','year','citation_count']]
     .drop_duplicates()
     .dropna()
     .sort_values('citation_count',ascending=False)
     .reset_index(drop=True)
    )
    author_cite_per_year['per_year_rank']=\
    author_cite_per_year.groupby('year').citation_count.rank(ascending=False,pct=True)
    author_cite_per_year =\
        author_cite_df.merge(author_cite_per_year)
    

    
    author_full = (author_sub.merge(
            author_gender[['author_id','reported_gender','perceived_gender']]
         )
        .merge(submissions[['submission_id','year']])
        .merge(ins_per_year.drop(['per_year_cnt','ins_total_cnt'],axis=1), on=['submission_id','author_id','year'],how='left')

        .merge(author_cite_per_year[['author_id',
                                     'submission_id',
                   'citation_count','per_year_rank', 'ins_adj_cnt']], on=['submission_id','author_id'],
              how='left').rename({'citation_count':'author_year_citation',
                                  'per_year_rank':'author_year_rank',
                                  'ins_adj_cnt': 'inst_year_csranking_adj',
                                 },axis=1)
    )
    
    author_full = author_full.merge(sub_arx_time[['submission_id','arxiv_first']], how='left')

    author_full.perceived_gender=author_full.perceived_gender.replace({None:np.nan})
    author_full.perceived_gender=author_full.perceived_gender.astype(float)
    author_full['reported_f'] = author_full.reported_gender.apply(
        lambda x : 1 if x=='Female' else 0)
    author_full['perceived_f'] = author_full.perceived_gender.apply(
        lambda x : np.nan if pd.isna(x) else (1 if x<0.3 else 0) )
    author_full['top_inst'] = author_full['ins_per_year_rank_pct'].apply(
        lambda x : x <= 0.01 )
    author_full['top_author'] = author_full['author_year_rank'].apply(
        lambda x : x <= 0.01 )
    return author_full



def get_author_full_pivot(get_table_handle,**kwargs):
    author_full = get_table_handle('author_full')
    max_author_no = author_full.n_author.max()
    dummy_df = pd.DataFrame([[-1]*len(author_full.columns)]*(max_author_no)
                    ,columns=author_full.columns
                    )
    dummy_df['submission_id'] = "###"
    dummy_df['author_no'] = np.arange(1,max_author_no+1)
    
    author_full_pivot = pd.pivot_table(
    pd.concat([author_full,dummy_df]), values=[
        'author_year_citation', 'author_year_rank',
        'ins_per_year_rank', 'ins_cum_cnt', 'inst_year_csranking_adj',
        'top_inst', 'top_author', 'reported_gender', 'perceived_gender','iso_a3'
                   ], 
            index=['submission_id', 'n_author'],
            columns=['author_no'], 
            fill_value=np.nan
    )
    
    author_full_pivot.columns = ['_'.join([str(c) for c in col]).strip()
        for col in author_full_pivot.columns.values]
    
    author_full_pivot=author_full_pivot.reset_index()
    author_full_pivot=author_full_pivot[~(author_full_pivot.submission_id=='###')].reset_index()
    return author_full_pivot



    
def get_group_inst_per_year(get_table_handle, *, grp_col='iso_a3', **kwargs):
    grp_cols = [grp_col] if not isinstance(grp_col, list) else grp_col
    domain_per_year = get_table_handle('ins_per_year')
    domain_per_year['per_year_cnt'] = (domain_per_year
     .groupby(grp_cols+['year'])
     .submission_id
     .transform('nunique')
    )

    domain_per_year=\
    domain_per_year[grp_cols+['year','per_year_cnt']].drop_duplicates()
    domain_per_year=domain_per_year.sort_values('per_year_cnt',ascending=False).dropna()
    domain_per_year['total_cnt'] =\
        domain_per_year.groupby(grp_cols).per_year_cnt.transform('sum')
    domain_total = (domain_per_year[grp_cols+['total_cnt']]
                 .drop_duplicates()
                 .sort_values('total_cnt', ascending=False)
                 .reset_index(drop=True)
                )
    return domain_per_year

def get_inst_ranking(get_table_handle, **kwargs):  
    """
    decision: all or Accept or Reject
    """
    decisions = get_table_handle('decision')
    submissions = get_table_handle('submissions')
    author_sub = get_table_handle('author_submission')
    author_inst_year_explode = get_table_handle('author_inst_year_explode')
    inst_domains = get_table_handle('institution_domains')
    
    
    ins_per_year = (submissions[['submission_id','year']]
     .merge(decisions[['forum','decision']].rename(
             {'forum':'submission_id'},axis=1))
     .merge(author_sub,how='left')
     .merge(author_inst_year_explode[['author_id','year','ins_root','ins_domain','tld']],
            how='left',on=['author_id','year']
           )
     .replace({'decision': {
        'Accept (Poster)':'Accept',
        'Accept (Spotlight)':'Accept',
        'Accept (Oral)':'Accept',
        'Accept (Talk)':'Accept',
        'Invite to Workshop Track':'Reject',}

        })
      .query("decision=='Accept'")
    )[['submission_id','year', 'ins_root',]].drop_duplicates()

    ins_per_year['per_year_cnt'] = (ins_per_year
     .groupby(['ins_root','year'])
     .submission_id
     .transform('nunique')
    )
    
    ins_per_year=ins_per_year.drop('submission_id',axis=1).drop_duplicates()
    ins_per_year=ins_per_year.sort_values('per_year_cnt',ascending=False).dropna()
    ins_per_year['ins_total_cnt'] =\
        ins_per_year.groupby(['ins_root']).per_year_cnt.transform('sum')

    ins_per_year = ins_per_year.sort_values(['ins_root','year'],ascending=True)
    ins_per_year['ins_cum_cnt'] =\
        ins_per_year.groupby(['ins_root',]).per_year_cnt.transform('cumsum')

    ins_per_year['ins_per_year_rank']=\
        ins_per_year.groupby('year').ins_cum_cnt.rank(ascending=False,pct=False)

    ins_per_year['ins_per_year_rank_pct']=\
        ins_per_year.groupby('year').ins_cum_cnt.rank(ascending=False,pct=True)
    ins_per_year_rank = ins_per_year
    ins_per_year_rank.year=ins_per_year_rank.year.astype(float).astype(int)
    


    inst_domains['iso_a3'] = inst_domains['alpha_two_code'].apply(
            lambda s : s if (s is None or pycountry.countries.get(alpha_2=s) is None) else pycountry.countries.get(alpha_2=s).alpha_3
        )
    inst_domains=inst_domains[~(inst_domains.iso_a3.str.len()!=3)].reset_index(drop=True)

    ins_per_year = (submissions[['submission_id','year']]
     .merge(decisions[['forum','decision']].rename(
             {'forum':'submission_id'},axis=1))
     .merge(author_sub,how='left')
     .merge(author_inst_year_explode[['author_id','year','ins_root','ins_domain','tld']],
            how='left',on=['author_id','year']
           )
     .merge(
        inst_domains[['domain','iso_a3','country']], left_on='ins_domain', right_on='domain', how='outer'
     )
     .assign(full_decision=lambda x:x['decision'])
     .replace({'decision': {
        'Accept (Poster)':'Accept',
        'Accept (Spotlight)':'Accept',
        'Accept (Oral)':'Accept',
        'Accept (Talk)':'Accept',
        'Invite to Workshop Track':'Reject',}

        })
    )[['submission_id','author_id','year', 'full_decision','decision', 'ins_root','iso_a3','tld', 'country']].drop_duplicates()

    ins_per_year=ins_per_year.dropna(subset=['submission_id'])
    ins_per_year = ins_per_year.merge(ins_per_year_rank,on=['ins_root','year'])
    ins_per_year['year'] = ins_per_year['year'].astype(int)
    
    return ins_per_year


def get_author_inst_year_explode(get_table_handle,**kwargs):
    """
    table dependencies: N/A
    """
    def reindex_by_date(grp):
        return grp.reindex(list(range(2016,2023))).ffill()

    author_inst = get_table_handle('author_institution')
    author_inst = author_inst.replace({'start':{pd.NA: '2016'}})
    author_inst.start =author_inst.start.astype(int)

    
    author_inst_year = ( author_inst[['author_id',
                                      'start',
                                      'ins_domain',
                                      'tld',
                                      'institution_id',
                                      'ins_root']].rename({
            'start':'year'},axis=1)
       .drop_duplicates(['author_id','year'])
       .sort_values(['author_id','year'])
       .reset_index(drop=True)
    )
    
    author_inst_year['nearest_year']=\
     author_inst_year.groupby('author_id').year.transform(
        lambda y: y[y<=2016].max() if (y<=2016).any() else 0
    )

    author_inst_year.loc[author_inst_year.year==author_inst_year.nearest_year,
                        'year'] = 2016
    
    author_inst_year_explode = (author_inst_year
     .set_index(['year'])
     .groupby('author_id')
     .apply(reindex_by_date)
    ).drop(['author_id'],axis=1).reset_index().drop(['nearest_year'],axis=1)
    return author_inst_year_explode



def get_review_agg(get_table_handle,**kwargs):
    """
    table dependencies: sub_reviews
    """
    sub_reviews = get_table_handle('sub_reviews')
   
    sub_reviews_agg=(sub_reviews
     .assign(rating_avg=lambda x:x.rating_int,
             rating_max=lambda x:x.rating_int,
             rating_min=lambda x:x.rating_int,
             confidence_avg=lambda x:x.confidence_int,
             confidence_max=lambda x:x.confidence_int,
             confidence_min=lambda x:x.confidence_int,
             sentiment_avg=lambda x:x.sentiment,
             sentiment_max=lambda x:x.sentiment,
             sentiment_min=lambda x:x.sentiment,
             rlen_avg=lambda x:x.review_len,
             rlen_min=lambda x:x.review_len,
             rlen_max=lambda x:x.review_len,
            )
     .groupby('submission_id')
     .agg({
         'year':'first', 'full_decision':'first', 'binary_decision':'first','input_len':'first',
         'n_review':'first',
         'rating_avg':'mean', 'rating_max':'max', 'rating_min':'min',
         'confidence_avg':'mean', 'confidence_max':'max', 'confidence_min':'min',
         'sentiment_avg':'mean', 'sentiment_max':'max', 'sentiment_min':'min',
         'rlen_avg':'mean', 'rlen_max':'max', 'rlen_min':'min',
     })
    ).reset_index() 
    return sub_reviews_agg

def get_review_plt_data(get_table_handle,**kwargs):
    pass

def get_review_data(get_table_handle, **kwargs):
    """
    table dependencies: N/A
    """
    reviews_len = get_table_handle('review_len', )
    submission_len = get_table_handle('submission_len', )
    reviews_special = get_table_handle('review_special_assessments', )
    reviews = get_table_handle('reviews', )
    decisions = get_table_handle('decision', )
    submissions = get_table_handle('submissions', )
    comment_sentiment = get_table_handle('comment_sentiment', )
    
#     pd.read_sql("select * from iclr_dataset;", con=filedb_con)
    reviews_len=utils.df_col_totype(reviews_len, ['review_len'], int)
    submission_len=utils.df_col_totype(submission_len, ['input_len'], int)
    
    reviews= reviews.merge(
        (reviews_special
            .query("conf_name=='ICLR2020'")
            .query("category=='experience_assessment'")
             [['id','assessment']]
            ), on='id', how='outer'
        ).replace({'assessment':{
           'I have published in this field for several years.':5,
           'I have published one or two papers in this area.':3.75,
            'I have read many papers in this area.':2.25,
           'I do not know much about this area.':1,
    }})
    reviews.loc[reviews.confidence_int.isna(),'confidence_int'] =\
           reviews.loc[reviews.confidence_int.isna(),'assessment']
    
    sub_reviews = (reviews[['id','forum','rating_int', 'confidence_int']]
     .rename({'forum':'submission_id'},axis=1)
     .merge(submissions[['submission_id','year']])
     .merge((comment_sentiment[['id','rbt_sentiment_mean']]
             .rename({'rbt_sentiment_mean':'sentiment'},axis=1)
        ),how='left')
     .merge(reviews_len[['review_id','review_len']],
           left_on=['id'],right_on=['review_id'],how='left')
     .merge(submission_len[['submission_id','input_len']],
           left_on=['submission_id'],right_on=['submission_id'],how='left')
     .merge(decisions[['forum','decision']].rename(
            {'forum':'submission_id'},axis=1
         ),how='left')
     .assign(binary_decision=lambda x : x['decision'])
     .replace({'binary_decision':{
         'Accept (Poster)':1,
         'Reject':0, 
         'Accept (Spotlight)': 1,
         'Accept (Oral)':1,
           'Accept (Talk)':1,
         'Invite to Workshop Track':0, 
     }})
     .rename({'decision': 'full_decision'}, axis=1)

    )

    sub_reviews['n_review'] = (sub_reviews
        .groupby("submission_id")
        .review_id
        .transform('nunique')
    )
    
    utils.df_col_totype(sub_reviews,['rating_int','confidence_int',
                                'sentiment','input_len','review_len'],float)
    
    sub_reviews.loc[(sub_reviews.year==2017)&(sub_reviews.confidence_int.isna()),'confidence_int']=\
        sub_reviews.query("year==2017").confidence_int.mean()
    
    sub_reviews['review_type'] = sub_reviews['rating_int'].apply(lambda s : 'Positive' if s > 7 else ('Negative' if s < 5 else 'Borderline'))
 
    
    return sub_reviews




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    