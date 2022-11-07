import os
from bs4 import BeautifulSoup, NavigableString
def parse_authors(article):
    """
    Parse authors from a given BeautifulSoup of an article
    """
    author_names = article.find('sourcedesc').findAll('persname')
    authors = []
    for author in author_names:
        firstname = author.find('forename', {'type': 'first'})
        firstname = firstname.text.strip() if firstname is not None else ''
        middlename = author.find('forename', {'type': 'middle'})
        middlename = middlename.text.strip() if middlename is not None else ''
        lastname = author.find('surname')
        lastname = lastname.text.strip() if lastname is not None else ''
        if middlename != '':
            authors.append(firstname + ' ' + middlename + ' ' + lastname)
        else:
            authors.append(firstname + ' ' + lastname)
    authors = '; '.join(authors)
    return authors


def parse_date(article):
    """
        Parse date from a given BeautifulSoup of an article
    """
    pub_date = article.find('publicationstmt')
    year = pub_date.find('date')
    year = year.attrs.get('when') if year is not None else ''
    return year


def parse_abstract(article):
    """
    Parse abstract from a given BeautifulSoup of an article 
    """
    div = article.find('abstract')
    abstract = ''
    for p in list(div.children):
        if not isinstance(p, NavigableString) and len(list(p)) > 0:
            abstract += ' '.join([elem.text for elem in p if not isinstance(elem, NavigableString)])    
    return abstract


def calculate_number_of_references(div):
    """
    For a given section, calculate number of references made in the section
    """
    n_publication_ref = len([ref for ref in div.find_all('ref') if ref.attrs.get('type') == 'bibr'])
    n_figure_ref = len([ref for ref in div.find_all('ref') if ref.attrs.get('type') == 'figure'])
    return {
        'n_publication_ref': n_publication_ref,
        'n_figure_ref': n_figure_ref
    }


def parse_sections(article, as_list=False):
    """
    Parse list of sections from a given BeautifulSoup of an article 
    
    Parameters
    ==========
    as_list: bool, if True, output text as a list of paragraph instead
        of joining it together as one single text
    """
    article_text = article.find('text')
    divs = article_text.find_all('div', attrs={'xmlns': 'http://www.tei-c.org/ns/1.0'})
    sections = []
    for div in divs:
        div_list = list(div.children)
        if len(div_list) == 0:
            heading = ''
            text = ''
        elif len(div_list) == 1:
            if isinstance(div_list[0], NavigableString):
                heading = str(div_list[0])
                text = ''
            else:
                heading = ''
                text = div_list[0].text
        else:
            text = []
            heading = div_list[0]
            if isinstance(heading, NavigableString):
                heading = str(heading)
                p_all = list(div.children)[1:]
            else:
                heading = ''
                p_all = list(div.children)
            for p in p_all:
                if p is not None:
                    try:
                        text.append(p.text)
                    except:
                        pass
            if not as_list:
                text = '\n'.join(text)
        if heading != '' or text != '':
            ref_dict = calculate_number_of_references(div)
            sections.append({
                'heading': heading,
                'text': text,
                'n_publication_ref': ref_dict['n_publication_ref'],
                'n_figure_ref': ref_dict['n_figure_ref']
            })
    return sections


def parse_references(article):
    """
    Parse list of references from a given BeautifulSoup of an article
    """
    reference_list = []
    references = article.find('text').find('div', attrs={'type': 'references'})
    references = references.find_all('biblstruct') if references is not None else []
    reference_list = []
    for reference in references:
        title = reference.find('title', attrs={'level': 'a'})
        if title is None:
            title = reference.find('title', attrs={'level': 'm'})
        title = title.text if title is not None else ''
        journal = reference.find('title', attrs={'level': 'j'})
        journal = journal.text if journal is not None else ''
        if journal == '':
            journal = reference.find('publisher')
            journal = journal.text if journal is not None else ''
        year = reference.find('date')
        year = year.attrs.get('when') if year is not None else ''
        authors = []
        for author in reference.find_all('author'):
            firstname = author.find('forename', {'type': 'first'})
            firstname = firstname.text.strip() if firstname is not None else ''
            middlename = author.find('forename', {'type': 'middle'})
            middlename = middlename.text.strip() if middlename is not None else ''
            lastname = author.find('surname')
            lastname = lastname.text.strip() if lastname is not None else ''
            if middlename != '':
                authors.append(firstname + ' ' + middlename + ' ' + lastname)
            else:
                authors.append(firstname + ' ' + lastname)
        authors = '; '.join(authors)
        reference_list.append({
            'title': title,
            'journal': journal,
            'year': year,
            'authors': authors
        })
    return reference_list


def parse_figure_caption(article):
    """
    Parse list of figures/tables from a given BeautifulSoup of an article
    """
    figures_list = []
    figures = article.find_all('figure')
    for figure in figures:
        figure_type = figure.attrs.get('type') or ''
        figure_id = '' if ('xml:id' not in figure.attrs) else figure.attrs['xml:id']

        label = figure.find('label').text
        if figure_type == 'table':
            caption = figure.find('figdesc').text
            data = figure.table.text
        else:
            caption = figure.text
            data = ''
        if figure_id == '' and figure_type == '' and data == '' and label=='' and caption=='':
            
            continue
        figures_list.append({
            'figure_label': label,
            'figure_type': figure_type,
            'figure_id': figure_id,
            'figure_caption': caption,
            'figure_data': data
        })
    return figures_list


def convert_article_soup_to_dict(article, as_list=False):
    """
    Function to convert BeautifulSoup to JSON format 
    similar to the output from https://github.com/allenai/science-parse/

    Parameters
    ==========
    article: BeautifulSoup

    Output
    ======
    article_json: dict, parsed dictionary of a given article in the following format
        {
            'title': ..., 
            'abstract': ..., 
            'sections': [
                {'heading': ..., 'text': ...}, 
                {'heading': ..., 'text': ...},
                ...
            ],
            'references': [
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...}, 
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...},
                ...
            ], 
            'figures': [
                {'figure_label': ..., 'figure_type': ..., 'figure_id': ..., 'figure_caption': ..., 'figure_data': ...},
                ...
            ]
        }
    """
    article_dict = {}
    if article is not None:
        title = article.find('title', attrs={'type': 'main'})
        title = title.text.strip() if title is not None else ''
        article_dict['authors'] = parse_authors(article)
        article_dict['pub_date'] = parse_date(article)
        article_dict['title'] = title
        article_dict['abstract'] = parse_abstract(article)
        article_dict['sections'] = parse_sections(article, as_list=as_list)
        article_dict['references'] = parse_references(article)
        article_dict['figures'] = parse_figure_caption(article)

        doi = article.find('idno', attrs={'type': 'DOI'})
        doi = doi.text if doi is not None else ''
        article_dict['doi'] = doi

        return article_dict
    else:
        return None


def parse_pdf_to_dict(parsed_article, as_list=False):
    """
    Parse the given 

    Parameters
    ==========
    pdf_path: str, path to publication or article

    Ouput
    =====
    article_dict: dict, dictionary of an article
    """
    article_dict = convert_article_soup_to_dict(parsed_article, as_list=as_list)
    return article_dict
