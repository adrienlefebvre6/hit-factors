from hit_factors.text import join_key, normalize_artist, normalize_title, split_artists


def test_normalize_title_retire_versions_et_featurings():
    assert normalize_title("Blinding Lights - Remastered 2020") == "blinding lights"
    assert normalize_title("Old Town Road (feat. Billy Ray Cyrus)") == "old town road"
    assert normalize_title("Déjà Vu (Radio Edit)") == "deja vu"


def test_split_artists():
    assert split_artists("Drake Featuring Rihanna & Future") == ["Drake", "Rihanna", "Future"]
    assert split_artists("Lil Nas X") == ["Lil Nas X"]


def test_normalize_artist_garde_l_artiste_principal():
    assert normalize_artist("Beyoncé Featuring JAY-Z") == "beyonce"


def test_join_key_identique_entre_sources():
    billboard = join_key("Old Town Road", "Lil Nas X Featuring Billy Ray Cyrus")
    spotify = join_key("Old Town Road (feat. Billy Ray Cyrus) - Remix", "Lil Nas X, Billy Ray Cyrus")
    assert billboard == spotify == "old town road|lil nas x"


def test_split_artists_ne_coupe_pas_les_noms_composes():
    assert split_artists("Post Malone x Swae Lee") == ["Post Malone", "Swae Lee"]
    assert split_artists("Florence + The Machine") == ["Florence + The Machine"]
