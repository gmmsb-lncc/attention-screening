# Lições Aprendidas — Otimização da Arquitetura DT-Kinase v7

*Registro científico do processo de diagnóstico e tentativas de melhoria
sobre a arquitetura `cross-attention 2D + CNN + HierPool` no contexto da
matriz cross-dataset 3×3 do benchmark `semantic-screening`.*

---

## 1. Contexto experimental

A tese de doutorado em curso defende o paradigma de *semantic screening*,
isto é, a previsão de bioatividade entre kinases e ligantes a partir
exclusivamente de notações lineares (sequência de aminoácidos e SMILES),
sem recorrer a estruturas tridimensionais ou descritores químicos manuais.
A arquitetura **DT-Kinase v7** materializa essa proposta combinando dois
PLMs congelados (ESM-2 8M para proteína; MoLFormer para ligante), um par
de adaptadores residuais com inicialização-identidade, oito mapas de
interação 2D obtidos por projeção em espaço de baixa dimensão seguida
de produto interno, uma rede convolucional bidimensional de quatro camadas
com convolução dilatada, e um *pool* hierárquico de atenção em dois
estágios cuja saída alimenta um classificador linear.

A construção da matriz cross-dataset 3×3 (treinamento em `human`,
`non_human` ou `all`; avaliação no mesmo conjunto e nos demais) revelou
que, embora competitiva intra-corpus, a arquitetura DT-Kinase v7 ocupava
a terceira posição dentre quatro modelos no agregado fora da diagonal.
A tabela abaixo sumariza esse cenário, com as médias calculadas sobre as
seis transferências cross-corpus disponíveis e cinco sementes por célula.

| Modelo            | MCC off-diag médio | Posição |
|-------------------|-------------------:|:-------:|
| DrugBAN           | 0,348              |    1    |
| GraphBAN          | 0,342              |    2    |
| **DT-Kinase v7**  | **0,298**          |  **3**  |
| ConPLex           | 0,209              |    4    |

A diferença de 0,050 MCC entre DT-Kinase v7 e o líder DrugBAN motivou a
investigação descrita neste documento.

---

## 2. Hipóteses sobre a origem da diferença

Antes de qualquer modificação na rede, foi conduzida uma análise
arquitetural comparativa dos quatro modelos avaliados, visando identificar
os componentes responsáveis pela diferença de desempenho. Quatro
hipóteses emergiram, ordenadas pela força do argumento e custo da
intervenção corretiva.

A primeira hipótese — e a mais robustamente sustentada pela teoria — é
que o gargalo de projeção do v7 destrói informação. Os mapas de interação
de v7 são construídos projetando a sequência de proteína de
$D_p = 320$ dimensões e a de ligante de $D_l = 768$ dimensões para um
espaço comum de apenas $d_h = 32$ dimensões antes do produto escalar.
Esse colapso, equivalente a passar pelos *bottlenecks* sequencialmente,
descarta informação que tanto o DrugBAN quanto o GraphBAN preservam ao
operar sobre uma matriz bilinear $W_k \in \mathbb{R}^{D_p \times D_l}$,
calculando $M_k = P\, W_k\, L^\top$ sem qualquer projeção intermediária.
A segunda hipótese diz respeito ao viés indutivo da CNN bidimensional:
ao tratar o mapa de interação como uma imagem, a v7 implicitamente assume
localidade — duas posições próximas no mapa têm interações relacionadas
— premissa que não é claramente justificável em pares proteína-ligante,
onde qualquer resíduo pode interagir com qualquer subestrutura.
A terceira hipótese aponta para o viés taxonômico herdado dos PLMs
congelados: ESM-2 foi pré-treinado em todo o UniProt, transportando
vieses de distribuição que podem prejudicar separação fina dentro do
clade restrito das kinases. A quarta hipótese, finalmente, observa a
ausência de qualquer componente de adaptação de domínio em v7,
contrastando com o uso explícito de CDAN (Conditional Domain Adversarial
Network) no DrugBAN.

A inspeção do código-fonte revelou um achado importante: o `variant=v8`
da `InteractionMapCNN` já implementa o cabeçote BAN bilinear, com tensor
$W_\mathrm{ban}$ de forma $(K, D_p, D_l)$ inicializado por Xavier,
acrescentando aproximadamente 1,96M parâmetros à rede. Esse caminho
nunca havia sido executado sob o regime numérico atual (fp32 puro com
TF32 desabilitado), o que sugeriu uma troca de configuração de baixo
custo como ponto de partida natural.

---

## 3. Reformulação do objetivo

A discussão técnica seguiu uma direção fértil até o momento em que se
reconheceu que o objetivo da tese não é vencer DrugBAN no MCC absoluto.
A contribuição científica em jogo é a validação do paradigma de
*semantic screening* — predizer a partir de 1D sem recorrer a estruturas
3D — e não a reprodução de uma arquitetura concorrente. Adotar BAN, CDAN
ou *knowledge distillation* indiscriminadamente significaria tornar
DT-Kinase indistinguível de DrugBAN ou GraphBAN, diluindo precisamente
o que justifica o capítulo metodológico da tese.

A partir desse esclarecimento, o objetivo foi redefinido: melhorar a
capacidade de classificação **preservando a identidade arquitetural**
do v7, isto é, mantendo o uso de *cross-attention* 2D por produto
escalar, a CNN bidimensional dilatada, o *HierPool* de atenção e o
regime de *backbones* congelados. Apenas escalonamentos de capacidade
e refinamentos numéricos seriam considerados; substituições estruturais
(como BAN ou *knowledge distillation*) ficaram fora do escopo dessa
fase.

---

## 4. Tier A — escalonamento de capacidade ("v7-plus")

A primeira intervenção, batizada `v7+`, agregou quatro mudanças de
configuração simultaneamente, todas atuando sobre componentes já
existentes na arquitetura. O número de cabeças de interação foi dobrado
de oito para dezesseis, oferecendo ao CNN um tensor de entrada com mais
canais distintos. A dimensão por cabeça foi elevada de 32 para 64,
reduzindo o gargalo de projeção apontado na primeira hipótese sem
descaracterizar o produto escalar como mecanismo de pontuação. O
classificador linear final foi substituído por um MLP de dois estágios
com não-linearidade GELU intermediária, conforme a opção
`mlp_head: true` já presente no código. Por fim, o adaptador residual
teve seus *bottlenecks* duplicados (de 256 para 512 no ramo proteico e
de 512 para 1024 no ligante), oferecendo maior capacidade de refinamento
das representações congeladas pelos PLMs.

A primeira execução desse conjunto, ainda mantendo `patience=5` e
`lr_mult=5.0` herdados do v7, produziu resultado pior que a linha base.

| Configuração                   | Train MCC | Test MCC |
|--------------------------------|----------:|---------:|
| v7 (linha base, seed 42)       |    0,5208 |   0,4862 |
| v7+ (Tier A, patience=5, lr_mult=5) | 0,5115 |   0,4697 |

A queda simultânea no treino e no teste afastou a hipótese de
*overfitting* clássico e apontou para *underfitting* por convergência
incompleta. Duas causas mecânicas foram identificadas. Primeiro, o
mesmo `patience=5` que se mostrava adequado para v7 (com cerca de
4 milhões de parâmetros treináveis) era prematuro para v7+ (com
8-10 milhões), provocando *early stop* antes da convergência completa.
Segundo, o multiplicador diferencial de taxa de aprendizado do
adaptador, `lr_mult=5.0`, foi calibrado para um adaptador menor; com
parâmetros dobrados, a taxa efetiva de $5 \times 10^{-4}$ ultrapassava
a estabilidade ótima.

Após a correção desses dois fatores (`patience: 15` e `lr_mult: 2.0`),
a execução repetida produziu o ganho esperado, fechando aproximadamente
um terço da distância original em direção ao alvo.

| Configuração                          | Train MCC | Test MCC | Δ vs v7 |
|---------------------------------------|----------:|---------:|--------:|
| v7 (linha base)                       |    0,5208 |   0,4862 |       — |
| v7+ Tier A — patience=5, lr\_mult=5   |    0,5115 |   0,4697 |  −0,016 |
| **v7+ Tier A — patience=15, lr\_mult=2** | **0,5652** | **0,5004** | **+0,014** |

A diferença entre treino (0,565) e teste (0,500), de aproximadamente
0,065 MCC, é consistente com a faixa observada para v7 e indica
generalização saudável; o ganho de 0,014 MCC sobre a linha base, embora
modesto, validou a estratégia de escalonamento para os componentes
escolhidos. O alvo definido pelo orientador para esta fase, contudo,
permanecia em $\mathrm{MCC} \geq 0,52$, exigindo intervenções adicionais.

---

## 5. Tier B — *pool* multi-cabeça e a importância da inicialização

A segunda intervenção atacou o componente de *pooling*: o
`_AxisAttentionPool` original utiliza uma única *query* aprendível por
estágio, oferecendo uma capacidade extremamente limitada de extrair
representações diversas do mapa de *features* convolucional. A
modificação proposta — chamada Tier B — elevou esse número para quatro
*queries* por estágio, com os vetores resultantes concatenados ao longo
da dimensão das *features* e projetados de volta para a dimensionalidade
do CNN por meio de uma camada linear `head_proj`.

Esperava-se um ganho adicional de 0,01 a 0,02 MCC. O resultado observado,
contudo, foi uma regressão acentuada.

| Configuração                          | Train MCC | Test MCC |   Δ |
|---------------------------------------|----------:|---------:|----:|
| v7+ Tier A (patience=15, lr\_mult=2)  |    0,5652 |   0,5004 |   — |
| v7+ Tier A + Tier B (pool=4 cabeças)  |    0,5200 |   0,4560 | −0,044 |

A queda simultânea no treino (0,565 → 0,520) e no teste (0,500 → 0,456)
descartou novamente a hipótese de *overfitting*. A análise do problema
revelou que a versão de *pool* com uma única *query*, herdada do v7,
opera no instante inicial do treinamento em um regime próximo à
identidade: a única *query*, multiplicada pela função *softmax* sobre
posições mascaradas, gera um vetor médio aproximadamente uniforme das
*features* CNN, comportamento que o resto da rede tinha aprendido a
explorar. A versão multi-cabeça, ao concatenar quatro saídas e
projetá-las por uma camada linear inicializada por Xavier (sem
inicialização-zero), produz logo no primeiro *forward* uma combinação
linear aleatória dos vetores de atenção, descaracterizando o regime
inicial em que o resto da rede dependia. O efeito é análogo ao
observado anteriormente nas injeções *side-feature* do v8 POC: capacidade
adicional sem inicialização-identidade desestabiliza componentes
adjacentes que não foram treinados para tolerar tais perturbações.

A lição metodológica é importante e merece destaque. Toda mudança que
inserir uma transformação aprendível sobre o caminho de informação
preexistente deve ser projetada para reduzir-se a uma identidade no
instante $t = 0$, transferindo ao gradiente a responsabilidade de
ativar a nova capacidade somente quando ela demonstrar utilidade
comprovada. No caso específico do *pool* multi-cabeça, a correção
natural seria zerar os pesos da `head_proj` ou substituir a concatenação
seguida de projeção por uma simples média sobre as quatro saídas (que
se reduz a um caso degenerado da projeção quando as cabeças são
equivalentes).

---

## 6. Tier C — perda contrastiva auxiliar

Após o reverso do Tier B (`pool_num_heads` retornado a um), a próxima
estratégia testada foi a inclusão de uma perda auxiliar contrastiva
inspirada no ConPLex, controlada pelos parâmetros `contrastive_weight`,
`cosine_feat` e `contrastive_dim`, todos já implementados no código mas
desativados por padrão. O mecanismo projeta os vetores de proteína e
ligante pós-*pool* para uma esfera de 128 dimensões e adiciona ao
*loss* total um termo que aproxima pares positivos e afasta negativos,
ponderado por um fator de 0,3. Adicionalmente, a similaridade-cosseno
entre os dois vetores projetados é concatenada como *feature* escalar à
entrada do classificador (`cosine_feat: true`).

A motivação é dupla. Do ponto de vista de regularização, a perda
contrastiva força a rede a aprender uma estrutura geométrica
significativa no espaço de *features* poolado, o que pressionar contra
memorização específica do *corpus*. Do ponto de vista de transferência,
a estrutura aprendida é mais robusta a *distribution shift*, dado que
ângulos no espaço unitário tendem a ser preservados ao longo de
distribuições relacionadas (uma propriedade explorada com sucesso no
ConPLex).

A execução desse experimento estava em curso no momento da redação deste
documento; o resultado será integrado em revisão posterior.

---

## 7. Síntese das observações

A trajetória experimental conduzida até aqui ofereceu três lições de
caráter geral que transcendem o problema específico da arquitetura
DT-Kinase. A primeira é que escalonamento de capacidade não pode ser
desacoplado dos hiperparâmetros de otimização: dobrar o tamanho do
adaptador exige reduzir o multiplicador de taxa de aprendizado
proporcionalmente, e aumentar o número de parâmetros treináveis exige
estender o orçamento de épocas (`patience`) para que a rede tenha
oportunidade de convergir. A primeira execução do Tier A falhou
precisamente por ignorar essa interdependência; sua correção, sem
qualquer mudança arquitetural, produziu o ganho desejado.

A segunda lição diz respeito à preservação do regime inicial. Toda
adição de capacidade que introduzir uma transformação aprendível
inicializada aleatoriamente sobre o caminho de informação preexistente
quebra a continuidade de treinamento que o resto da rede aprendera a
explorar. O Tier B exemplifica esse princípio: a transição de uma para
quatro *queries* no *pool* não é em si problemática — é a projeção
linear `Linear(4D, D)` com inicialização Xavier que o é. O remédio, em
geral, é inicializar a saída da nova transformação com pesos zerados
para que a rede comece exatamente como antes da modificação e
gradativamente ative a nova capacidade conforme o gradiente o requeira.

A terceira lição é metodológica. Iterar com uma única semente é
defensável apenas para diagnóstico rápido durante o desenvolvimento da
arquitetura; para concluir cientificamente sobre a eficácia de uma
mudança, é imprescindível repetir o experimento com cinco sementes e
reportar média acompanhada de desvio-padrão. A diferença observada
entre os dois primeiros experimentos do Tier A (0,470 e 0,500) excede
o desvio-padrão típico observado na linha base v7 (aproximadamente
0,02), confirmando que a tendência identificada não é mero ruído de
inicialização. Mudanças cuja diferença sobre v7 fica abaixo desse
limiar não devem ser interpretadas como benefício antes da validação
multi-semente.

A tabela abaixo consolida o progresso observado até este ponto.

| Etapa                          | Mudança                                | Train MCC | Test MCC | Δ vs v7 |
|--------------------------------|----------------------------------------|----------:|---------:|--------:|
| v7 baseline                    | linha base, seed 42, NH                | 0,5208    | 0,4862   |     —   |
| Tier A não-tunado              | heads=16, head_dim=64, MLP, adapter 2× | 0,5115    | 0,4697   | −0,016  |
| **Tier A tunado**              | acima + patience=15, lr_mult=2         | **0,5652**| **0,5004**| **+0,014** |
| Tier B (sobre Tier A)          | + pool_num_heads=4                     | 0,5200    | 0,4560   | −0,044  |
| Tier C (sobre Tier A)          | + contrastive_weight=0,3, cosine_feat  | *em execução* | *em execução* | *—* |

---

## 8. Observações sobre o ambiente computacional

Um aspecto que merece registro, embora não diretamente relacionado à
arquitetura, é a importância do regime numérico. As execuções iniciais
em `diamante-02` revelaram incompatibilidade entre o cuDNN 9.x do
ambiente e o *driver* CUDA 12.4 do hospedeiro: a convolução bidimensional
em precisão dupla (`float64`) abortava com `CUDNN_STATUS_NOT_INITIALIZED`,
forçando a transição para `float32`. Posteriormente, descobriu-se que o
TF32 ativo nas operações de multiplicação matricial truncava a
mantissa do `float32` a aproximadamente dez bits, comprometendo a
precisão na profunda pilha de atenção do modelo. A desativação explícita
do TF32 no caminho `pure_fp32` (controlada automaticamente pela função
`build_optim` em `level4_cnn.py`) restaurou a qualidade numérica e
permitiu que v7 atingisse o desempenho esperado de aproximadamente
0,486 MCC na semente 42.

Em `diamante-01`, com cuDNN saudável, o regime computacional foi
mantido em `float32` por questão de tempo (a precisão dupla incorre em
penalidade de aproximadamente duas vezes em tempo), porém com cuDNN
ativo e TF32 desativado. Essa configuração — `float32` com mantissa
completa e operações cuDNN otimizadas — combina precisão numérica e
velocidade aceitável, e é a recomendada para experimentos subsequentes
nesta linha de pesquisa.

---

## 9. Considerações finais e próximos passos

A jornada documentada neste registro confirma que a arquitetura
DT-Kinase v7 é um sistema bem balanceado dentro de seu próprio espaço
de configuração: ganhos significativos exigem ou correção cuidadosa
dos hiperparâmetros de otimização ao escalonar capacidade, ou
intervenções pontuais em componentes que não afetem a continuidade do
regime inicial de treinamento. Tentativas indiscriminadas de aumentar
expressividade — como o *pool* multi-cabeça do Tier B — podem
prejudicar a rede se não respeitarem essa continuidade.

O alvo de $\mathrm{MCC} \geq 0,52$ permanece em aberto. As três
possibilidades mais promissoras para fechá-lo, em ordem decrescente de
risco-retorno, são: (i) confirmar o resultado do Tier C em execução,
e se positivo, combinar Tier A com a perda contrastiva como
configuração de referência; (ii) investigar uma versão corrigida do
*pool* multi-cabeça com inicialização-zero da projeção `head_proj`,
preservando a identidade do v7 no instante inicial; e (iii) considerar
LoRA sobre as duas camadas superiores dos PLMs ESM-2 e MoLFormer, o
que relaxaria a hipótese mais profunda — a do viés taxonômico — sem
substituir o paradigma de *backbones* congelados característico da
metodologia proposta.

Independentemente do caminho escolhido, a próxima etapa deve incluir
necessariamente a confirmação multi-semente do melhor protótipo,
reportando $\mathrm{MCC} \pm \sigma$ sobre as cinco sementes canônicas
do protocolo da tese.

---

## Referências

Bai *et al.*, "Interpretable bilinear attention network with domain
adaptation improves drug-target prediction", *Nature Machine Intelligence*
(2023).

Singh *et al.*, "Contrastive learning in protein language space predicts
interactions between drugs and protein targets", *PNAS* (2023).

Zhang *et al.*, "GraphBAN: An inductive framework for drug-target
interaction using knowledge distillation", *Nature Communications*
(2024).

Hu *et al.*, "LoRA: Low-rank adaptation of large language models",
*ICLR* (2022).

Chicco & Jurman, "The advantages of the Matthews correlation coefficient
(MCC) over F1 score and accuracy", *BMC Genomics* (2020).
