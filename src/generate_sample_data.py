"""
generate_sample_data.py
Creates a synthetic Fake/Real news dataset for testing when the ISOT Kaggle
dataset is not yet downloaded.  Produces True.csv and Fake.csv in ../data/.
"""

import random
import pandas as pd
import os

random.seed(42)

# ─── Real news templates ────────────────────────────────────────────────────
REAL_TITLES = [
    "Federal Reserve raises interest rates by 0.25 percent amid inflation concerns",
    "Senate passes bipartisan infrastructure bill with $1.2 trillion in funding",
    "WHO declares end of COVID-19 global health emergency after three years",
    "Scientists discover new exoplanet in habitable zone using James Webb telescope",
    "Apple unveils new iPhone model with improved battery life and camera system",
    "Stock markets close higher as inflation data shows signs of cooling",
    "European Union reaches agreement on new climate policy framework",
    "NASA successfully launches Artemis mission to the Moon",
    "Study finds Mediterranean diet reduces risk of heart disease by 30 percent",
    "G7 leaders agree on new sanctions package targeting energy sector",
    "University researchers develop breakthrough battery technology",
    "Congress approves emergency funding for natural disaster relief",
    "Tech giant announces 10,000 layoffs amid economic slowdown",
    "Supreme Court issues ruling on landmark environmental case",
    "United Nations climate summit concludes with new emissions pledges",
    "Central bank holds interest rates steady amid mixed economic signals",
    "Pharmaceutical company receives FDA approval for new cancer treatment",
    "International trade agreement signed between EU and Southeast Asian nations",
    "New study links air pollution to increased dementia risk in elderly",
    "City council approves $500 million affordable housing development plan",
]

REAL_TEXTS = [
    "Washington — The Federal Reserve on Wednesday raised its benchmark interest rate "
    "by a quarter of a percentage point, the ninth increase in its current tightening "
    "cycle, as policymakers continue their effort to bring inflation back down to the "
    "2 percent target. Fed Chair Jerome Powell said at a press conference that the "
    "committee remains committed to returning inflation to its goal, though he acknowledged "
    "the process has been gradual. The decision was unanimous among voting members. "
    "Markets reacted relatively calmly, with the S&P 500 closing up 0.3 percent.",

    "Washington — The Senate passed a $1.2 trillion infrastructure bill with broad "
    "bipartisan support on Tuesday, marking a significant legislative achievement. "
    "The legislation allocates funding for roads, bridges, broadband internet expansion, "
    "and the electrical grid. President Biden called the passage a historic investment "
    "in America's future. The bill now heads to the House for a final vote. "
    "Transportation Secretary Pete Buttigieg said implementation would begin within months.",

    "Geneva — The World Health Organization on Friday officially declared the end of "
    "COVID-19 as a public health emergency of international concern, more than three "
    "years after the designation was first made. WHO Director-General Tedros Adhanom "
    "Ghebreyesus said the virus remains a global threat but that countries now have the "
    "tools and knowledge to manage it. More than 6.9 million deaths have been officially "
    "reported to WHO, though the true toll is estimated to be considerably higher.",

    "The James Webb Space Telescope has confirmed the existence of a rocky exoplanet "
    "located within the habitable zone of its host star, approximately 40 light-years "
    "from Earth. The findings, published in the journal Nature, suggest the planet "
    "could potentially harbor liquid water on its surface. The discovery was made by "
    "an international team of astronomers who analyzed infrared light data collected "
    "over several months. Scientists cautioned that further observations are needed to "
    "determine whether the planet has an atmosphere.",
]

REAL_SUBJECTS = ["politics", "government news", "worldnews", "News", "US_News"]

# ─── Fake news templates ────────────────────────────────────────────────────
FAKE_TITLES = [
    "BREAKING!!! Deep State Globalists PANIC as Trump Prepares to EXPOSE Everything!!!",
    "SHOCKING TRUTH: Bill Gates ADMITS vaccines contain microchips to control population!!!",
    "WAKE UP SHEEPLE: The moon landing was STAGED and NASA has been LYING for 50 YEARS!!",
    "BOMBSHELL: Scientists CONFIRM 5G towers are activating mind-control frequencies NOW!!",
    "The REAL reason they want to ban cash – total surveillance of EVERY purchase you make!",
    "EXPOSED: George Soros funding secret plan to replace American citizens with immigrants!!",
    "DOCTORS HATE HIM: Man cures cancer in 3 days using this ONE WEIRD TRICK!!!",
    "SECRET MEETING reveals elite plans to reduce world population by 90 percent by 2030!!",
    "MUST SHARE BEFORE IT'S DELETED: Government putting fluoride in water to DUMB DOWN population!!",
    "SUPPRESSED TECHNOLOGY: Free energy machine inventor MURDERED by Big Oil cartels!!!",
    "ALERT: New law will allow government to seize your bank account WITHOUT warning!!!",
    "BANNED VIDEO: What they don't want you to know about chemtrails and weather control!!",
    "CLINTON CRIME FAMILY exposed in new documents – mainstream media BLACKOUT!!!",
    "URGENT: They are putting nanobots in your food – here's how to protect yourself!!!",
    "The TRUTH about the Federal Reserve – it's not federal and has NO reserves!!!",
    "Hollywood elite caught in massive satanic ritual – this will BLOW YOUR MIND!!!",
    "WHISTLEBLOWER reveals CIA has been running drug trade since the 1980s!!!",
    "PROOF: COVID was engineered in a lab and the cure was ready BEFORE the outbreak!!!",
    "BREAKING: Rothschilds control ALL central banks – the conspiracy is REAL!!!",
    "EXPOSED: The REAL reason they're pushing electric cars – total control of your movement!!",
]

FAKE_TEXTS = [
    "A high-ranking WHISTLEBLOWER within the deep state has come forward with BOMBSHELL "
    "evidence that will DESTROY the narrative being pushed by the globalist elites and "
    "their puppets in the mainstream media!!! The documents, which the fake news is REFUSING "
    "to cover, prove beyond any doubt that the entire operation has been orchestrated from "
    "the very top. SHARE THIS EVERYWHERE before it gets taken down!!! They cannot silence "
    "the TRUTH much longer. Patriots are WAKING UP and the elite are PANICKING!!! "
    "Do your own research and don't believe what the government WANTS you to think!!!",

    "AMAZING!!! A patriot insider who worked at the highest levels of the government for "
    "20 years has LEAKED documents proving what many have suspected for years. The "
    "mainstream media will NEVER report on this because they are CONTROLLED by the same "
    "forces trying to enslave humanity. Wake up AMERICA!!! The time to act is NOW before "
    "it is too late!!! This information is being SUPPRESSED across all major platforms. "
    "A MASSIVE cover-up is underway. Forward this to everyone you know!!! "
    "THEY FEAR AN INFORMED CITIZENRY. This is being DELETED everywhere – SAVE IT NOW!!!",

    "What the GOVERNMENT doesn't want you to know is finally being revealed by brave "
    "insiders who risk their lives to bring you the TRUTH!!! Declassified documents obtained "
    "through sources we cannot name for their protection CONFIRM what whistleblowers have "
    "been saying for decades. Big Pharma, the deep state, and their allies in Hollywood "
    "are ALL involved in this SINISTER plot against ordinary Americans. "
    "Do NOT comply! Do NOT submit! The REVOLUTION of truth begins NOW!!! "
    "Your government LIES to you every single day. WAKE UP before it's too late!!! "
    "Share this video before YouTube BANS it in 24 hours!!!",
]

FAKE_SUBJECTS = ["News", "politics", "left-news", "Government News", "US_News", "Middle-east"]


def generate_dataset(n_real=5000, n_fake=5000):
    """Generate synthetic dataset."""
    real_rows, fake_rows = [], []

    for i in range(n_real):
        real_rows.append({
            "title"  : random.choice(REAL_TITLES) + f" ({2019 + (i % 5)})",
            "text"   : " ".join(random.choices(REAL_TEXTS, k=2)),
            "subject": random.choice(REAL_SUBJECTS),
            "date"   : f"{random.randint(2016,2021)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        })

    for i in range(n_fake):
        fake_rows.append({
            "title"  : random.choice(FAKE_TITLES) + f" ({2019 + (i % 5)})",
            "text"   : " ".join(random.choices(FAKE_TEXTS, k=2)),
            "subject": random.choice(FAKE_SUBJECTS),
            "date"   : f"{random.randint(2016,2021)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        })

    return pd.DataFrame(real_rows), pd.DataFrame(fake_rows)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    real_path = os.path.join(out_dir, "True.csv")
    fake_path = os.path.join(out_dir, "Fake.csv")

    if os.path.exists(real_path) and os.path.exists(fake_path):
        print("✅ True.csv and Fake.csv already exist – skipping generation.")
        print("   (Delete them and re-run to regenerate, or replace with the real Kaggle dataset)")
    else:
        print("Generating synthetic dataset …")
        real_df, fake_df = generate_dataset(n_real=5000, n_fake=5000)
        real_df.to_csv(real_path, index=False)
        fake_df.to_csv(fake_path, index=False)
        print(f"✅ True.csv  → {len(real_df):,} rows  ({real_path})")
        print(f"✅ Fake.csv  → {len(fake_df):,} rows  ({fake_path})")
        print("\nTip: Replace these synthetic files with the real Kaggle ISOT dataset")
        print("     for better model accuracy on your final submission.")
