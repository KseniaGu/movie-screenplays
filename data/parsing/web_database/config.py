from enum import Enum


class WebDatabaseConfig:
    sources_names = Enum(
        'sources', 'imsdb scriptorama simplyscripts gotothehistory dailyscript awesomefilm slug horror scriptologist '
                   'sfy scifi dailyactor screenplays_online thescriptlab onscreen selling reddit')
    scripts_dir = 'data/screenplay_data/data/raw_texts'
    scriptorama = {
        'table_links': (
            'http://www.script-o-rama.com/table.shtml',
            'http://www.script-o-rama.com/table2.shtml',
            'http://www.script-o-rama.com/table3.shtml',
            'http://www.script-o-rama.com/table4.shtml'
        )
    }
    simplyscripts = {
        'page_path': 'parsing/web_database/data/simplyscripts_page.html'
    }
    gotothehistory = {
        'page_path': 'https://gointothestory.blcklst.com/script-download-links-9313356d361c'
    }
    dailyscript = {
        'page_path': 'parsing/web_database/data/daily_page_1.html',
        'source_url': 'https://www.dailyscript.com/'
    }
    awesomefilm = {
        'source_url': 'https://www.dailyscript.com/'
    }
