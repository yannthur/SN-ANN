import yfinance as yf

def download_nflx_to_csv(filename="Data/nflx_data.csv"):
    """
    Télécharge l'historique complet de NFLX et le sauvegarde en CSV.
    """
    print(f"Extraction des données Netflix depuis Yahoo Finance...")
    
    ticker = yf.Ticker("NFLX")
    df = ticker.history(period="max")
    
    if df.empty:
        print("Erreur : Impossible de récupérer les données.")
        return
    df.reset_index(inplace=True)
    
    # Sauvegarde
    df.to_csv(filename, index=False)
    print(f"✅ Succès ! {len(df)} lignes sauvegardées dans : {filename}")

if __name__ == "__main__":
    download_nflx_to_csv()