from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

load_dotenv()

model = os.getenv("LLM_MODEL")


def main():
    print("Hello from langchain-course!")
    information = """Elon Reeve Musk[1] (AFI:[ˈiːlɒn ˈɹiːv ˈmʌsk]) (Pretoria, 28 giugno 1971) è un imprenditore e politico sudafricano con cittadinanza canadese naturalizzato statunitense.

    Ricopre i ruoli di fondatore, amministratore delegato e direttore tecnico della compagnia aerospaziale SpaceX,[2] fondatore di The Boring Company[3] e della società di intelligenza artificiale xAI, cofondatore di Neuralink e OpenAI,[4] amministratore delegato e product architect della multinazionale automobilistica Tesla,[5] proprietario e presidente di X (precedentemente Twitter).[6] Ha inoltre proposto un sistema di trasporto superveloce conosciuto come Hyperloop One, posta in liquidazione il 21 dicembre 2023.[7] Tramite SpaceX gestisce Starlink, una costellazione satellitare che avrebbe l'obiettivo di fornire Internet ad alta velocità e bassa latenza a tutto il pianeta.[8]

    Secondo Forbes, al 7 febbraio 2026, con un patrimonio stimato di 849,3 miliardi di dollari, risulta essere la persona più ricca del mondo.[9][10][11]

    Dal 20 gennaio al 29 maggio 2025 è stato a capo del Dipartimento dell'Efficienza Governativa statunitense.[12][13][14]

    Biografia
    Famiglia d'origine

    Elon Musk mentre osserva i resti del Falcon 9R nel 2014
    Suo padre, Errol Musk, è cittadino sudafricano ma di ascendenza britannica[15]. Ingegnere elettromeccanico, pilota e navigatore ora in pensione, già co-proprietario di una miniera di smeraldi in Zambia.[16][17] La madre, Maye Haldeman[18] canadese[16] ha lavorato come dietologa e modella[19].

    Maye è figlia di Joshua Norman Haldeman (1902-1974), politico canadese che ha attirato l'attenzione della stampa mondiale negli anni 2020: nato negli USA[20] ma emigrato giovanissimo in Canada con la famiglia[21], Joshua perse la casa durante la carestia degli anni 1930[22]; divenne un attivista del movimento tecnocratico[22] che teorizzava la sostituzione del governo canadese da parte di ingegneri non eletti, anche con l'uso della forza[22]. Condannato dai tribunali canadesi per infedeltà alla corona britannica[22][23], Haldeman si candidò in seguito alla carica di primo ministro[22]. Difese la pubblicazione dei Protocolli dei Savi di Sion, famoso falso storico, sostenendo che il fatto che non fossero autentici "non era il punto"[22] perché provavano comunque il ruolo degli ebrei come élite mondiale. Per lo stesso motivo, fu un acceso sostenitore dell'apartheid Sudafricano[22] in quanto lo vedeva come "reazione della civiltà bianca cristiana" contro la "cospirazione internazionale"[22] ordita da "bande di neri" e dai banchieri ebrei[24]. Appassionato di aerei, era solito viaggiare con la moglie e le due figlie piccole in un aereo monopropulsore in tela[22]. Si trasferì in Sudafrica nel 1949, seguendo la profezia di un medium[22]. Una volta in Sudafrica, si appassionò alle rovine in muratura presenti nella zona e dedicò anni di ricerche nel deserto del Kalahari per trovare una città perduta, che secondo lui sarebbe stata lasciata da una misteriosa civiltà bianca ormai scomparsa[22]. Poche settimane dopo il massacro di Sharpeville, in cui una manifestazione contro l'apartheid vide la polizia sudafricana sparare sui dimostranti, Haldeman pubblicò un volumetto di 48 pagine, intitolato The International Conspiracy to Establish a World Dictatorship and its Menace to South Africa ("La cospirazione internazionale che ambisce a creare una dittatura mondiale e la sua minaccia per il Sudafrica")[22]. A metà degli anni 1960 pubblicò The International Conspiracy in Health ("La cospirazione internazionale nella sanità"), dove spiegava che una cospirazione internazionale, di matrice comunista, controllava le banche, i media, le università e spingeva per l'obbligo di vaccinazione, di pastorizzazione del latte e di fluorizzazione dell'acqua[22]. Haldeman morì a 72 anni in un incidente aereo[22].

    Infanzia
    Elon Musk è nato il 28 giugno 1971 a"""

    summary_template = """ given the {information} about Elon Musk, write a concise summary of his biography in Italian."""

    summary_prompt = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGroq(model=model, temperature=0)

    chain = summary_prompt | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
