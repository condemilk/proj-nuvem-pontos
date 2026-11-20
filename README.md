# Inspeção geométrica de nuvens de pontos — MVP

Aplicação para carregar uma nuvem de pontos (`.xyz`, `.csv`, `.pcd` ou `.ply`) de uma parede,
informar manualmente a largura e a altura de referência, e comparar com o que foi medido:
largura, altura e desaprumo. O sistema registra o histórico em SQLite e exporta para CSV.

Esta versão não usa modelos BIM/IFC, dados de demonstração, simulação ao vivo nem qualquer
modo de alinhamento de pose: a referência é só a largura e a altura informadas pelo usuário,
e a orientação da parede (eixos horizontal/vertical, desaprumo) é obtida diretamente da normal
do plano ajustado por RANSAC sobre a nuvem.

## Executar

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abra `http://127.0.0.1:8000`. Faça upload de uma nuvem, escolha a unidade, preencha largura e
altura de referência e clique em **Analisar nuvem**.

Para executar os testes: `python -m pytest tests` (22 testes).

## Estrutura

- `app/geometry/`: `preprocess` (limpeza), `planes` (RANSAC), `measures` (largura, altura,
  desaprumo), `deviation` (aplicação de tolerâncias), `analyze` (pipeline de inspeção),
  `synthetic` (gerador de nuvens sintéticas para os testes automatizados — sem nenhuma
  simulação de pose/posição de sensor; a parede sintética já nasce no referencial esperado
  pelo pipeline, com o sensor implicitamente na origem).
- `app/ingest/point_cloud.py`: leitura e validação de arquivos de nuvem.
- `app/db/repository.py`: persistência SQLite, histórico e exportação CSV.
- `app/main.py`: API FastAPI.
- `app/static/`: interface web e visualizador 3D.

## Premissa de orientação (importante)

O sistema **não alinha a nuvem a nenhuma referência externa** nem recebe pose do sensor.
Ele assume que o eixo Z da nuvem recebida já é a vertical real — ou seja, **o sensor precisa
estar nivelado** (ou a nuvem já corrigida para isso) antes da captura. A partir disso:

- os eixos horizontal/vertical da parede são derivados automaticamente da normal do plano
  ajustado por RANSAC e desse Z global (não é preciso informar nada sobre a pose do sensor);
- largura, altura e desaprumo são invariantes à posição do sensor e ao giro em torno do eixo Z
  (yaw) — só o nivelamento (roll/pitch) importa;
- se o sensor estiver fora de nível, o desaprumo medido fica viesado por essa inclinação, sem
  forma de o software corrigir isso sem alguma informação externa. Esse comportamento está
  coberto pelo teste `test_sensor_must_be_level_or_plumb_is_biased`.

## Observações sobre o processamento

- As unidades internas são metros. Selecione corretamente a unidade do arquivo no upload.
- O limiar RANSAC é informado em milímetros na interface; define a distância máxima de um
  ponto ao plano para ele ser considerado parte do plano dominante.
- O algoritmo exige pelo menos 30 pontos compatíveis com o plano dominante. Se não atingir esse
  mínimo, a mensagem de erro informa quantos pontos foram encontrados e sugere verificações.
- Largura e altura são medidas sobre os pontos **brutos** (antes do filtro estatístico) que
  caem sobre o plano ajustado — o filtro de outliers corrói bordas em varreduras de grade
  regular e encurtaria as duas dimensões.
- Se a nuvem for uma varredura em um único plano (uma linha, sem o segundo eixo), altura e
  desaprumo ficam "n/d" e um aviso é emitido.
- Se o plano dominante não for predominantemente vertical (ex.: piso/teto escaneado por
  engano), um aviso é emitido; o desaprumo ainda é calculado, mas pode não ser significativo.
- Os valores de tolerância iniciais (10 mm de dimensão, 5 mm/m de desaprumo) são provisórios;
  defina os do projeto ou da norma aplicável.

## Banco de dados e migração

O esquema atual tem só as colunas usadas hoje (sem `mode`, sem `bim_file`). Ao abrir um banco
criado por uma versão anterior do protótipo (que tinha os modos `best_fit`/`pose` e referência
por BIM), `repo.init_db()`:
- renomeia a coluna `bim` de `measurements` para `reference_value`, se necessário;
- remove as colunas obsoletas `mode` e `bim_file` de `scans` via `ALTER TABLE ... DROP COLUMN`
  (suportado a partir do SQLite 3.35); em um SQLite mais antigo, a remoção falha silenciosamente
  e a coluna só deixa de ser usada, sem provocar erro.

Nenhuma linha de `scans` ou `measurements` é apagada nessa migração — ver
`test_db_migration_drops_mode_column_without_losing_rows`.

## Histórico e arquivos

Os resultados são armazenados em `data/inspecoes.db`; os arquivos enviados ficam em
`data/uploads/`. A exportação CSV reúne uma linha por medida de cada inspeção (3 por
inspeção: largura, altura e desaprumo). A interface mostra, na seção "Histórico", só os
botões **Baixar CSV** e **Imprimir relatório**; consultas por medida individual continuam
disponíveis via API (`GET /api/history?key=plumb`, `GET /api/scans`) para uso por linha de
comando ou uma futura tela, e são usadas diretamente pelos testes de banco de dados.
