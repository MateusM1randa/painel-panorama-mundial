# Painel: Panorama mundial (saúde, renda e população)

Painel interativo com visões coordenadas (Streamlit + Plotly), dados Gapminder 1952-2007.

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar (Streamlit Community Cloud, gratuito)

1. Crie um repositório público no GitHub e envie estes arquivos, mantendo a estrutura:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
2. Acesse https://share.streamlit.io e entre com sua conta do GitHub.
3. Clique em **Create app**, escolha o repositório, a branch (`main`) e o arquivo principal `app.py`.
4. Clique em **Deploy**. O endereço público sai no formato `https://<nome>.streamlit.app`.

## Como usar o painel

- Barra lateral: ano, indicador (mapa e ranking), continentes e botão "Limpar seleção".
- Clique em um país no mapa ou em uma barra do ranking para selecioná-lo.
- No gráfico de dispersão, arraste uma caixa ou laço para selecionar vários países.
- Todas as visões destacam a seleção e a tabela final lista só os países selecionados.
