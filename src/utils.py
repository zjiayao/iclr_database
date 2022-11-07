import numpy as np
import pandas as pd
import tqdm
import thefuzz
import thefuzz.fuzz
import ast
import openreview as oprev
import urllib
from bs4 import BeautifulSoup
import src.pdf_txt_parser as ptp
import arxiv
## DB interfacing

def get_log_col(df, cols):
    for col in cols:
        new_col = f"{col}_log"
        if new_col not in df.columns:
            df[new_col] = df[col].apply(np.log)
    return df


def drop_col(df, cols):
    return (df.drop(cols, axis=1, errors='ignore')
               .reset_index(drop=True)
              )

def get_table_cols(table_name, con):
    return pd.read_sql(f"select * from {table_name} limit 0;", con=con).columns

def get_table(table_name, con, to_drop=['index'], **kwargs):
    df = pd.read_sql(f"select * from {table_name};", con=con)
    to_drop = [col for col in to_drop if col in df.columns]
    return df.drop(to_drop, axis=1)

def get_table_filter(table_name, con, filter_col, filter_str, to_drop=['index']):
    df = pd.read_sql(f"select * from {table_name} where {filter_col} = {filter_str};", con=con)
    to_drop = [col for col in to_drop if col in df.columns]
    return df.drop(to_drop, axis=1)

def get_table_update_sql(table_name, row, id_col, columns):
    row_id = row[id_col]
    params = [f"{param}" if type(param) not in [int, float] else param for param in row[columns]] + [row_id]
    cols_and_fields = ','.join([f"{col} = ?" for col in columns])
    return (f"""update {table_name} set {cols_and_fields} where id = ?;""", params)

def add_cols_to_table(table_name, con, cols):
    cur = con.cursor()
    for col in cols:
        cur.execute(f"alter table {table_name} add column '{col}';")
        con.commit()
    cur.close()

def auto_add_cols_to_table(table_name, con, df, dry_run=False):
    new_cols_to_add = [col for col in df.columns if col not in
                  get_table_cols(table_name, con)]
    if dry_run:
        if len(new_cols_to_add) > 0:
            print(f"The folowing column(s) will be added to the table '{table_name}':\n", '\t'.join(new_cols_to_add))
        else:
            print(f"No new columns to be added")
    else:
        add_cols_to_table(table_name, con, new_cols_to_add)
    
    
def get_table_creation_sql(tbl_name, columns, pkey=None, fkeys=[], df=None):
    """
    fkey: (key, ref)
    """
    def kt(key):
        if df is None:
            return key

        if df[key].dtype is int:
            tp = 'INTEGER'
        elif df[key].dtype is float:
            tp = 'REAL'
        elif df[key].dtype is str:
            tp = 'TEXT'
        elif df[key].dtype is bin:
            tp = 'BLOB'
        else:
            tp = 'TEXT'
        return f"{key} {tp}"
    
    col_str = ','.join([kt(c) for c in columns])
    fkey_str = ''
    pkey_str = ''
    if pkey is not None:
        if not hasattr(pkey, '__iter__'):
            pkey_str = f",PRIMARY KEY ({pkey})"
        else:
            pkey_str = f",PRIMARY KEY ({','.join(pkey)})"
#     print(pkey, pkey_str)
    if len(fkeys) > 0:
        fkey_str = ','+','.join([f"FOREIGN KEY({key}) REFERENCES {ref}" for (key, ref) in fkeys])
        
    return f"""CREATE TABLE IF NOT EXISTS {tbl_name} ({col_str}{fkey_str}{pkey_str});"""

def create_table(con, tbl_name, columns, pkey, fkeys=[], df=None):
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute(get_table_creation_sql(tbl_name, columns, pkey, fkeys, df))
    con.commit()
    cur.close()
    
def insert_pd(con, tbl_name, df):
    df.to_sql(name=tbl_name, con=con, index=False, if_exists='append')

def create_table_and_insert(df, con, tbl_name, pkey, fkeys=[]):
    create_table(con, tbl_name, df.columns, pkey, fkeys, df)
    insert_pd(con, tbl_name, df)
    
    
# pd.DataFrame tools
def df_col_tostr(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda s : s.__repr__())
    
def df_col_totype(df, cols, tp):
    for col in cols:
        df[col] = df[col].astype(tp, errors='ignore')
    return df
        
def lit_eval_df(df, cols):
    def eval_no_exp(a):
        try:
            return ast.literal_eval(a)
        except:
            return a
    for col in cols:
        df[col] = df[col].apply(eval_no_exp)
# OpenReview paper manipulation

def paper_note_todf(note):
    note_json = note.to_json()
    note_dict = {k:v for k,v in note_json.items() if type(v) is not dict}
    for k,v in note_json.items():
        if type(v) is dict:
            note_dict.update(v)
    return pd.DataFrame.from_dict(note_dict, orient='index').T

def enum_inv_df(inv, client):
    notes_iter = oprev.tools.iterget_notes(client, invitation=inv)
    dfs = []
    for nidx, note in enumerate(notes_iter):
        try:
            df = paper_note_todf(note)
            dfs.append(df)
        except Exception as e:
            print(f"Exception at {nidx}: {e}")
    return pd.concat(dfs)

def enum_noteid_df(noteids, client):
    dfs = []
    for nidx, noteid in enumerate(noteids):
        try:
            df = paper_note_todf(client.get_note(noteid))
            dfs.append(df)
        except Exception as e:
            print(f"Exception at {nidx}: {e}")
    return pd.concat(dfs)

        
        
# OpenReview author profile manipulation
def unpack_history(hists):
    agg_hists = {
        'position': [],
        'ins_domain': [],
        'ins_name': [],
        'start': [],
        'end': [],
    }
    def _single_hist(hist):
        hist_cols = hist.keys()
        for k in ['position', 'start', 'end']:
            if k in hist_cols:
                agg_hists[k].append(hist[k])
            else:
                agg_hists[k].append(None)


        hist_inst_domain = None
        hist_inst_name = None
        if 'institution' in hist_cols:
            hist_inst = hist['institution']
            if 'domain' in hist_inst:
                hist_inst_domain = hist_inst['domain']
            if 'name' in hist_inst:
                hist_inst_name = hist_inst['name']
        agg_hists['ins_domain'].append(hist_inst_domain)
        agg_hists['ins_name'].append(hist_inst_name)
        
    _ = [_single_hist(hist) for hist in hists]
    df = pd.DataFrame(agg_hists)
    df['start'] = df['start'].astype(int, errors='ignore')
    df['end'] = df['end'].astype(int, errors='ignore')
    return df

def author_note_todf(note):
    note_json = note.to_json()
    note_dict = {k:v for k,v in note_json.items() if type(v) is not dict}
    for k,v in note_json.items():
        if k in ['metaContent']:
            continue
        if type(v) is dict:
            note_dict.update(v)
    
    if 'history' in note_dict:
        author_hist = note_dict['history']
        del note_dict['history']
        hists_df = unpack_history(author_hist)
        hists_df['author'] = note_dict['id']
    else:
        hists_df = None
    if 'relations' in note_dict:
        author_relations = note_dict['relations']
        del note_dict['relations']
    
#     print(note_dict, '\n', author_hist, '\n', author_relations)

    author_df = pd.DataFrame.from_dict(note_dict, orient='index').T
    cols_to_drop = ['referent', 'packaging', 'invitation','readers', 'nonreaders', 'signatures', 
                            'writers', 'active', 'password', 'expertise']
    cols_to_drop = [c for c in cols_to_drop if c in author_df.columns]
    author_df = author_df.drop(cols_to_drop, axis=1)
    try:
        name_dicts = author_df['names'].iloc[0]
        name_dict = name_dicts[0]
        for nd in name_dicts:
            if 'preferred' in nd and nd['preferred']:
                name_dict = nd
        
        author_df['name'] = name_dict['first'] + (' '+name_dict['middle']+' ' if len(name_dict['middle']) > 0 else ' ') +  name_dict['last']

    except Exception as e:
        print(f"Exception parsing names {author_df['names']}: {e}")
        pass
    author_df = author_df.drop(['names'], axis=1)

    return author_df, hists_df

def get_empty_author_df(author_id):
    return pd.DataFrame([[author_id]], columns=['id'])
def get_author_sub_df(author_id, sub_id, no):
    return pd.DataFrame([[author_id, sub_id, no]], columns=['author_id', 'submission_id', 'author_no'])


# GScholar Crawling
def get_field_from_url(u, fs, default=''):
    try:
        parse_res = urllib.parse.parse_qs(urllib.parse.urlparse(u).query)
        for f in fs:
            if f in parse_res:
                return parse_res[f][0]
    except Exception as e:
        print(f"Error parseing {u}: {e}")
    return default

def search_author_todf(s, author_parser):
    authorid = s['gs_author_id']
    res = author_parser.get_author(authorid)
    res = author_parser.fill(res, sections=['basics', 'indices', 'counts'], sortby='citedby', publication_limit=0)
    keep_col = [
        'name',
        'affiliation',
        'url_picture',
        'email_domain',
        'citedby',
        'citedby5y',
        'hindex',
        'hindex5y',
        'i10index',
        'i10index5y',
    ]
    author_dict = {}
    for k in keep_col:
        author_dict[k] = None
        if k in res:
            author_dict[k] = res[k]
    author_cites_df = None
    if 'cites_per_year' in res:
        cites_dict = res['cites_per_year']
        author_cites_df = pd.DataFrame({
            'author': [s['id']] * len(cites_dict),
            'year': [k for k in cites_dict],
            'cite': [cites_dict[k] for k in cites_dict],
        })
    author_df = pd.DataFrame.from_dict(author_dict, orient='index').T
    author_df['id'] = s['id']
    return author_df, author_cites_df
        

def search_author_partial_todf(s, res, author_parser):
#     res = author_parser.get_author(authorid)
    res = author_parser.fill(res, sections=['basics', 'indices', 'counts'], sortby='citedby', publication_limit=0)
    keep_col = [
        'name',
        'affiliation',
        'url_picture',
        'email_domain',
        'citedby',
        'citedby5y',
        'hindex',
        'hindex5y',
        'i10index',
        'i10index5y',
    ]
    author_dict = {}
    for k in keep_col:
        author_dict[k] = None
        if k in res:
            author_dict[k] = res[k]
    author_cites_df = None
    if 'cites_per_year' in res:
        cites_dict = res['cites_per_year']
        author_cites_df = pd.DataFrame({
            'author': [s['id']] * len(cites_dict),
            'year': [k for k in cites_dict],
            'cite': [cites_dict[k] for k in cites_dict],
        })
    author_df = pd.DataFrame.from_dict(author_dict, orient='index').T
    author_df['id'] = s['id']
    return author_df, author_cites_df
        
# computer arXiv similarity
def prepare_encode_str(s,title_col='title',abs_col='abstract'):
    return s[title_col] + '[SEP]' + s[abs_col]

def df_encode(df, smodel, batchsize=50, title_col='title', abs_col='abstract'):
    n = df.shape[0]
    n_batch = n // batchsize
    n_residue = n - n_batch*batchsize
    df['encode_str'] = df.apply(lambda s : prepare_encode_str(s, title_col, abs_col),
                               axis=1)

    ret_arrs = []
    for idx in tqdm.tqdm(range(n_batch)):

        ret_arrs.append( smodel.encode(df.iloc[idx*batchsize:(idx+1)*batchsize]['encode_str'].values) ) 
    if n_residue > 0:
        ret_arrs.append( smodel.encode(df.iloc[-n_residue:]['encode_str'].values) ) 
    return np.vstack(ret_arrs)

def fuzz_sim_short_txt(a, b):
    return thefuzz.fuzz.partial_ratio(a,b)/100

def df_compute_simlarity(s, submissions_df, vec_col):
    sub_id = s['submission_id']
    sub = submissions_df.query(f"id=='{sub_id}'")
    assert sub.shape[0] == 1
    sub = sub.iloc[0]
    d_author = len(sub.authors) - len(s.authors)
    v1, v2 = np.array(s[vec_col]), np.array(sub[vec_col])
    ip = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
    s['sim_author'] = fuzz_sim_short_txt(' '.join(sub.authors), ' '.join(s.authors))
    s['sim_author_cnt'] = d_author
    s['sim_contents_cos'] = ip
    return s
    
    
    
## pdf
def get_parsed_article_df(s):
    full_text = s['full_text']
    paper_id = s['id']
    parsed_article = BeautifulSoup(full_text, 'lxml')
    article_dict = ptp.parse_pdf_to_dict(parsed_article)

    paper_df = pd.DataFrame.from_dict({k:v for (k,v) in article_dict.items()
                                         if k not in 
                                         ['sections', 'references', 'figures']
                                        }, orient='index').T
    paper_df['id'] = paper_id
    paper_df[['n_fig', 'n_ref', 'n_sec']] = 0, 0, 0
    section_df, ref_df, fig_df = None, None, None
    if 'sections' in article_dict:
        section_df = pd.DataFrame(article_dict['sections'])
        section_df['id'] = paper_id
        paper_df['n_sec'] = section_df.shape[0]
    if 'references' in article_dict:
        ref_df = pd.DataFrame(article_dict['references'])
        ref_df['id'] = paper_id
        paper_df['n_ref'] = ref_df.shape[0]
    if 'figures' in article_dict:
        fig_df = pd.DataFrame(article_dict['figures'])
        fig_df['id'] = paper_id
        paper_df['n_fig'] = fig_df.shape[0]
        
    return paper_df, section_df, ref_df, fig_df
    
    
    
# arxiv
def arxiv_res_todf(res):
    attrs = ['authors',
     'categories',
     'comment',
     'doi',
     'entry_id',
     'journal_ref',
     'links',
     'pdf_url',
     'primary_category',
     'published',
     'summary',
     'title',
      'updated'
    ]
    res_dict = {at:getattr(res,at) for at in attrs}
    res_dict['authors'] = [a.name for a in res_dict['authors']]
    res_dict['links'] = [l.href for l in res_dict['links']]
    res_dict['id'] = res.get_short_id().split('v')[0]
    for k,v in res_dict.items():
        if v is None:
            if k in ['links', 'authors','categories']:
                res_dict[k] = []
            else:
                res_dict[k] = ''
    return pd.DataFrame(pd.Series(res_dict)).T

def arxiv_search_apply(sub_id, title, target_conf):
    search = arxiv.Search(
        query=title,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    res_df = pd.concat([arxiv_res_todf(r) for r in search.results()])
    res_df['submission_id'] = sub_id
    res_df['target_conf'] = target_conf
    df_col_tostr(res_df, ['authors','categories', 'links'])
    return res_df
    

