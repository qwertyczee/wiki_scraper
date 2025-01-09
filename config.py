# config.py
CATEGORIES = [
    # Věda a technika
    'Přírodní_vědy',
    'Fyzika',
    'Chemie',
    'Biologie',
    'Astronomie',
    'Matematika',
    'Geologie',
    'Informatika',
    'Inženýrství',
    'Medicína',
    'Technologie',
    
    # Společenské vědy
    'Historie',
    'Filozofie',
    'Psychologie',
    'Sociologie',
    'Ekonomie',
    'Politologie',
    'Právo',
    'Pedagogika',
    'Archeologie',
    'Antropologie',
    
    # Kultura a umění
    'Literatura',
    'Hudba',
    'Výtvarné_umění',
    'Film',
    'Divadlo',
    'Architektura',
    'Fotografie',
    'Tanec',
    'České_umění',
    'Světová_kultura',
    
    # Společnost
    'Náboženství',
    'Mytologie',
    'Společnost',
    'Politika',
    'Vzdělávání',
    'Média',
    'Žurnalistika',
    
    # Geografie a cestování
    'Geografie',
    'Česká_republika',
    'Evropa',
    'Světové_dějiny',
    'Cestování',
    'Města',
    'Hory',
    'Řeky',
    
    # Sport a zábava
    'Sport',
    'Olympijské_hry',
    'Fotbal',
    'Hokej',
    'Atletika',
    'Tenis',
    'Zimní_sporty',
    
    # Příroda
    'Zoologie',
    'Botanika',
    'Ekologie',
    'Ochrana_přírody',
    'Savci',
    'Ptáci',
    'Hmyz',
    'Rostliny',
    
    # Věda a výzkum
    'Vynálezy',
    'Objevy',
    'Vědecký_výzkum',
    'Kosmonautika',
    'Biotechnologie',
    'Nanotechnologie',
    
    # Každodenní život
    'Jídlo',
    'Nápoje',
    'Móda',
    'Životní_styl',
    'Volný_čas',
    'Domácnost',
    'Zahrada',
    
    # Technika a průmysl
    'Doprava',
    'Automobily',
    'Letectví',
    'Průmysl',
    'Stavebnictví',
    'Elektrotechnika',
    'Robotika'
]

# Zvýšení počtu článků na kategorii
MAX_ARTICLES_PER_CATEGORY = 2000
BATCH_SIZE = 100

# Přidání nové konstanty pro minimální délku článku
MIN_ARTICLE_LENGTH = 1000  # znaků