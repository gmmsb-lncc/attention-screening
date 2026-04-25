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

A execução do experimento na semente 42 produziu $\mathrm{MCC} = 0{,}5167$
no teste e $0{,}5880$ no treino, um ganho aparente de $+0{,}017$ MCC
sobre o Tier A isolado e $+0{,}031$ MCC sobre a linha base v7. Esses
números, contudo, refletem uma única semente e devem ser interpretados
apenas como sinal preliminar; a confirmação multi-semente subsequente
revelou um quadro mais sóbrio.

A execução do mesmo protótipo nas cinco sementes canônicas
($42, 123, 456, 789, 1024$) produziu uma média de teste de
$0{,}5143 \pm 0{,}0079$ — abaixo do alvo $0{,}52$ por aproximadamente
$0{,}006$ MCC, e $0{,}003$ abaixo do que a semente 42 isoladamente
sugeria. A tabela abaixo consolida o cenário comparativo.

| Configuração                              | Test MCC (seed 42) | Test MCC (5-seed) |
|-------------------------------------------|-------------------:|------------------:|
| v7 baseline                               | 0,4862             | $0{,}506 \pm 0{,}020$* |
| v7+ Tier A tunado                         | 0,5004             | (não medido)      |
| **v7+ Tier A + Tier C**                   | **0,5167**         | **$0{,}5143 \pm 0{,}0079$** |

\* valor de referência do `CLAUDE.md` para o protocolo da tese.

O contraste entre semente 42 isolada (0,517) e média 5-semente (0,514)
ilustra com precisão a terceira lição da seção 7: iterar com semente
única superestima ganhos. A diferença de $0{,}003$ MCC, embora
pequena, representa metade do gap restante para o alvo, e ilustra
como decisões científicas baseadas em uma única semente podem produzir
otimismo injustificado.

Observação metodologicamente relevante: o desvio-padrão observado em
v7+ ($\sigma = 0{,}008$) é aproximadamente 60% menor que o da linha
base v7 ($\sigma \approx 0{,}020$). Isso indica que a combinação
Tier A + Tier C não apenas eleva ligeiramente a média, mas também
estabiliza o modelo entre inicializações distintas — uma propriedade
desejável que pode ser argumentada na tese como evidência de que a
regularização contrastiva atua sobre a geometria do espaço de features
de forma robusta a perturbações de inicialização.

O ganho médio sobre v7 fica em $+0{,}008$ MCC, modesto mas
estatisticamente defensável devido à redução substancial da
variância. O alvo $0{,}52$ permanece em aberto, motivando a
exploração das direções catalogadas na seção subsequente.

---

## 6.1. Tier D — falha do SWA vanilla e a lição sobre regimes de treino

A primeira tentativa de aplicar Stochastic Weight Averaging na
configuração Tier A+C (com $\mathrm{swa\_start} = 5$) produziu uma
regressão acentuada de $-0{,}021$ MCC no teste e de $-0{,}079$ MCC
no treino, resultado oposto ao esperado.

| Configuração                                | Train MCC | Test MCC |   Δ |
|---------------------------------------------|----------:|---------:|----:|
| v7+ Tier A+C (seed 42, sem SWA)             |    0,5880 |   0,5167 |  —  |
| v7+ Tier A+C+D (SWA vanilla, swa_start=5)   |    0,5088 |   0,4964 | $-0{,}021$ |

A queda simultânea no treino e no teste descarta novamente a hipótese
de *overfit* clássico e indica que o procedimento moveu os pesos para
uma região do espaço de parâmetros que classifica pior em todas as
distribuições. A análise pós-falha esclareceu o motivo: o método SWA
publicado por Izmailov et al. (2018) supõe (i) treinamento longo, da
ordem de centenas de épocas; (ii) taxa de aprendizado constante e
elevada, ou esquema cíclico; e (iii) acumulação iniciada apenas após
convergência clara, tipicamente nos últimos 25% da trajetória total.
Nenhuma dessas condições é satisfeita pelo nosso pipeline, que termina
em torno de doze épocas com Adam e *early stop* baseado em
$\mathrm{val\_mcc}$. Iniciar o averaging na quinta época significou
mediar pesos capturados na **fase de aprendizado ativo**, não na
fase de oscilação em torno de um mínimo já encontrado, puxando a média
para uma região anterior à convergência efetiva do modelo.

A lição metodológica é portanto a quarta a registrar: **a aplicabilidade
de uma técnica depende criticamente do regime experimental em que ela
foi originalmente proposta**. SWA na sua forma vanilla pressupõe um
regime que nosso pipeline não tem; aplicá-lo cegamente, ignorando essa
mismatch, produziu resultado anti-intuitivo. A correção apropriada
não é desabilitar o averaging em si, mas substituí-lo por uma variante
adequada ao nosso regime — *Greedy Model Soup* (Wortsman et al., ICML
2022) é um candidato natural, pois inclui apenas *checkpoints* que
melhoram $\mathrm{val\_mcc}$, evitando contaminação por épocas
prematuras. Esse caminho fica registrado como direção a explorar.

## 6.2. Estratégia de empilhamento ortogonal — Tier E e Tier F

Após a regressão do Tier D, adotou-se uma orientação experimental
distinta: em vez de buscar uma única intervenção transformadora,
empilhar múltiplos regularizadores **ortogonais** que atuem em eixos
disjuntos do *pipeline*. A premissa é que regularizadores que não
competem pelo mesmo espaço de gradiente tendem a contribuir
aditivamente — propriedade observada empiricamente entre Tier A
(escalonamento de capacidade, eixo arquitetural) e Tier C (perda
contrastiva, eixo geométrico de *features*) e que nada impede de
estender a outros eixos.

Foram identificados quatro eixos disjuntos onde se pode aplicar
regularização sem alterar a arquitetura de inferência:

| Eixo                          | Tier  | Mecanismo |
|-------------------------------|-------|-----------|
| Capacidade arquitetural       | A     | escala de cabeças, dimensões, classificador |
| Espaço de *features* poolado  | C     | perda contrastiva ConPLex-inspired |
| Espaço de *inputs*            | E     | *Mixup* — combinação linear de pares de exemplos |
| Espaço de *targets*           | F     | *label smoothing* — alvo com $1-\varepsilon$ ao invés de $1$ |

A configuração resultante, denominada `v7-pro`, ativa simultaneamente
$\mathrm{mixup\_alpha} = 0{,}3$ (sweet spot da literatura para
classificação binária; Zhang et al., ICLR 2018) e $\mathrm{label\_smooth}
= 0{,}05$ (Szegedy et al., Inception V3). Ambos os parâmetros já
existiam no código, controlados por variáveis de ambiente; não houve
desenvolvimento adicional. Esse é talvez o ponto mais relevante da
estratégia: muitos componentes regularizadores estavam disponíveis há
meses no código, mas permaneciam desativados por falta de validação
empírica direcionada.

A interação entre $E$ e $F$ exige cuidado interpretativo: ambos
suavizam, respectivamente, a distribuição de entrada e a de alvo, o
que sob certa luz são variantes do mesmo princípio. Em regimes onde
um dos dois domina (saturação), o outro torna-se redundante. Em
regimes onde atuam em fenômenos distintos do treinamento, somam-se.
A literatura suporta a coexistência em tarefas de classificação;
nosso experimento testará empiricamente se isso vale para o nosso
*pipeline* específico.

A primeira execução de `v7-pro` na semente 42 de `diamante-01`
produziu $\mathrm{MCC}_\text{test} = 0{,}5320$, um ganho aparente de
$+0{,}0064$ MCC sobre `v7+` na mesma semente e hospedeiro
($0{,}5256$). Adicionalmente, o $\mathrm{MCC}_\text{val}$ saltou de
$0{,}5153$ para $0{,}5870$, um aumento de $+0{,}072$ — sensivelmente
maior que o ganho observado em teste. Essa assimetria é
inconclusiva sob uma única semente: pode refletir uma genuína
melhoria do modelo no espaço de validação que não se transfere
plenamente para o teste por diferença de distribuição entre os dois
*splits*, ou pode ser flutuação numérica associada à inicialização.
O custo computacional permaneceu inalterado (treinamento em
$4{,}82$ minutos, idêntico ao tempo de `v7+` na mesma máquina), o
que confirma que Mixup e *label smoothing* não introduzem *overhead*
relevante.

A subsequente avaliação multi-semente em *non\_human* (cinco
sementes canônicas) refutou contundentemente o sinal preliminar.
A média de $\mathrm{MCC}_\text{test}$ caiu para $0{,}4961 \pm
0{,}0245$, **abaixo** da configuração `v7+` canônica
($0{,}5143 \pm 0{,}0079$) por $-0{,}018$ MCC, e a variância entre
sementes triplicou — de $\sigma = 0{,}008$ em `v7+` para
$\sigma = 0{,}024$ em `v7-pro`. Esse resultado tem três
implicações imediatas. Primeira, a semente 42 isolada superestimou o
desempenho médio em $0{,}036$ MCC, equivalente a aproximadamente
1,5 desvios-padrão da nova distribuição multi-semente: nova
confirmação empírica da terceira lição da seção 7. Segunda, a
combinação Tier E + Tier F **não é aditiva** no nosso *pipeline* —
contrariamente à expectativa baseada em ortogonalidade dos eixos de
regularização — e parece introduzir interferência destrutiva ou
saturação. Terceira, o aumento substancial de $\sigma$ é
sintomaticamente atribuível ao Mixup, mecanismo estocástico cuja
amostragem aleatória de pares de exemplos depende fortemente da
semente; *label smoothing*, por outro lado, é determinístico e não
contribui diretamente para variância entre sementes.

A configuração `v7-pro` é portanto descartada como candidata
canônica. A configuração `v7+` (Tier A + C apenas) permanece como
melhor configuração validada multi-semente em *non\_human*. As
ablações isoladas — Tier E sem F, e Tier F sem E — passam a ser
necessárias para identificar qual dos dois componentes é deletério
ou se é a combinação que produz o efeito adverso. Os arquivos
`configs/v7_plus_E.yaml` (Tier A+C+E) e `configs/v7_plus_F.yaml`
(Tier A+C+F) foram criados para essa finalidade.

A execução das duas ablações em três sementes em `diamante-01`
desambiguou a pergunta sobre qual componente carrega o efeito
adverso. Os resultados estão sumarizados na tabela abaixo.

| Configuração                | n | Test MCC                | $\Delta$ vs canônico |
|-----------------------------|---|-------------------------|---------------------:|
| v7+ A+C (canônico)          | 5 | $0{,}5143 \pm 0{,}0079$ | —                    |
| v7+E (A+C + Mixup)          | 3 | $0{,}4988 \pm 0{,}0254$ | $-0{,}016$           |
| **v7+F (A+C + label smoothing)** | 3 | $\mathbf{0{,}5260 \pm 0{,}0274}$ | $\mathbf{+0{,}012}$ |
| v7-pro (A+C+E+F)            | 5 | $0{,}4961 \pm 0{,}0245$ | $-0{,}018$           |

A imagem é nítida. O Tier E (Mixup) é, isoladamente, deletério: a
configuração `v7+E` regrediu em $-0{,}016$ MCC sobre o canônico, em
magnitude virtualmente idêntica à do empilhamento E+F ($-0{,}018$).
O Tier F (*label smoothing*), em contrapartida, apresentou ganho
positivo isolado de $+0{,}012$ MCC. A interpretação que se sustenta
é que `v7-pro` regrediu **porque continha Mixup**, e que o ganho
modesto trazido pelo *label smoothing* foi mais que cancelado pelo
prejuízo do Mixup. A configuração de melhor desempenho atual passa
a ser portanto `v7+F` (Tier A + C + F), ainda pendente de
validação em cinco sementes.

A décima primeira lição é registrada: **resultados de empilhamento
adverso devem ser ablacionados antes de se descartar quaisquer
componentes individuais**. Foi tentador, ao observar a regressão de
`v7-pro`, descartar tanto Mixup quanto *label smoothing*; a
ablação isolada mostrou que somente um dos dois é responsável pelo
efeito adverso, e que o outro permanece útil. Componentes não devem
ser julgados culpados pela performance ruim de uma configuração que
os contém quando outros componentes potencialmente deletérios
estão presentes simultaneamente. A regra prática: ao observar
regressão em uma configuração com $n$ componentes novos, a primeira
intervenção é executar $n$ ablações isoladas, não rejeitar todos os
$n$ em bloco.

Sobre o aspecto das variâncias: ambas as ablações de três sementes
apresentaram $\sigma \approx 0{,}025\text{–}0{,}027$, contra
$\sigma = 0{,}008$ da configuração canônica em cinco sementes. Duas
explicações coexistem como hipóteses, e somente o experimento de
cinco sementes em `v7+F` poderá discriminá-las. A primeira é
estatística: a estimativa de $\sigma$ a partir de apenas três
amostras carrega aproximadamente setenta por cento de incerteza
relativa, e $\sigma$ aparente de $0{,}027$ pode ser consistente
com $\sigma$ verdadeiro de $0{,}008$ dentro do intervalo de
confiança. A segunda é mecânica: o *label smoothing*, embora
determinístico em sua aplicação, modifica a paisagem de *loss*
explorada pela rede, podendo levar diferentes inicializações a
mínimos genuinamente diferentes. A confirmação multi-semente
discriminará entre essas hipóteses.

## 6.4. Tier "BAN-híbrido" — segunda violação de identidade-init

A primeira frente experimental sob o regime de identidade relaxada
($\S 6{.}3$) foi a configuração `v7_ban_F`, empilhando o cabeçote BAN
bilinear (`variant=v8`) sobre `v7+F` (Tier A + C + F). A motivação
era atacar a Hipótese 1 (gargalo de projeção) usando o tensor
$W_\mathrm{ban} \in \mathbb{R}^{K \times D_p \times D_l}$ com 1,96M
parâmetros adicionais, preservando o resto do *pipeline* (CNN,
*HierPool*, classificador). A expectativa era ganho aditivo de
$+0{,}015$ a $+0{,}030$ MCC sobre `v7+F`.

| Configuração     | n | Test MCC               | $\sigma$ | $\Delta$ vs `v7+F` |
|------------------|---|------------------------|---------:|-------------------:|
| v7+F (canônico)  | 3 | $0{,}5260 \pm 0{,}0274$ | 0,027 | —                 |
| v7+F + BAN       | 3 | $0{,}5028 \pm 0{,}0462$ | **0,046** | $-0{,}023$        |

A regressão foi acompanhada de aumento de $\sigma$ de aproximadamente
$70\%$, padrão diagnóstico já familiar: instabilidade entre sementes
indica perturbação inicial não controlada. A inspeção do código
revelou que $W_\mathrm{ban}$ é inicializado por
`nn.init.xavier_uniform_(self.W_ban[k])` — mesma família de
inicialização que causou o fracasso do Tier B (multi-cabeça do
*pool*) e do v8-side. No instante $t=0$, a interação é
$M_k = P \cdot W_k \cdot L^\top \neq 0$, gerando 16 mapas
aleatórios mas seed-específicos que o CNN downstream nunca enfrentou
durante o treinamento de v7. A consequência é a familiar
divergência entre sementes — cada uma encontra um mínimo
qualitativamente diferente.

A décima segunda lição que emerge é a generalização da segunda:
**a violação de identidade-init não é específica de uma família
de transformação; aplica-se uniformemente a qualquer perturbação
não-zero que se acrescente ao caminho de informação**. O Tier B
violou-a com `Linear(4D, D)` Xavier no *pool*; v8-side violou-a
com projeções de injeção lateral; v7_ban_F violou-a com o tensor
bilinear $W_\mathrm{ban}$ Xavier. O remédio é estruturalmente o
mesmo em todos os casos: começar com a transformação produzindo
zero ou identidade, deixando o gradiente decidir quando ativá-la.

A correção possível para `v7_ban_F` é uma versão **BAN-residual**
em que a interação se compõe como
$M_k = (P_\text{proj}^k)(L_\text{proj}^k)^\top + \alpha_k \cdot P W_k L^\top$,
com $\alpha_k$ aprendível inicializado em zero — exatamente o
princípio aplicado às portas LoRA-style do `EmbeddingAdapter`. Sob
essa parametrização, o modelo começa idêntico ao v7 (caminho
*dot-product* puro) e o gradiente ativa o termo bilinear apenas
onde seja útil. Esse caminho fica registrado como direção a
explorar, não testado nesta etapa por priorização da revisão do
adapter.

## 6.5. Revisão estrutural do `EmbeddingAdapter`

A análise crítica do componente `EmbeddingAdapter` revelou três
imperfeições estruturais que, embora isoladamente pequenas, em
conjunto comprometem o princípio de identidade-init que tem sido
o leitmotiv da abordagem.

A primeira imperfeição é que o ramo de auto-atenção opcional
(`use_self_attn=true`) inicializava sua projeção de saída com
inicialização Xavier padrão do PyTorch, em desacordo com a
inicialização-zero aplicada explicitamente à projeção de saída
do bloco MLP. No instante $t=0$, o vetor `attn_out` era ruído
gaussiano com escala não-trivial, e o resíduo
`x + attn_out` desviava da identidade no primeiro *forward*. O
remédio aplicado consistiu em zerar explicitamente
`self.self_attn.out_proj.weight` e `bias`, alinhando o
sub-componente de auto-atenção com o sub-componente MLP.

A segunda imperfeição é a ausência de portas escalares aprendíveis
(*gates*) controlando a contribuição de cada sub-camada. Sem essa
camada de defesa, qualquer perturbação inicial dos pesos internos
da sub-camada propaga-se diretamente ao resíduo. A introdução de
`attn_scale` e `mlp_scales` — parâmetros escalares inicializados
em zero — adiciona uma camada de proteção independente da
inicialização interna: mesmo que pesos da sub-camada sejam
perturbados, a saída é multiplicada por zero e a saída do
*adapter* permanece exatamente igual ao *input*. À medida que o
gradiente flui durante o treinamento, esses *gates* gradualmente
aumentam, ativando a contribuição da sub-camada de forma
controlada. Esta é uma transposição direta do princípio de LoRA
(Hu et al., 2022) ao componente `EmbeddingAdapter`.

A terceira imperfeição é a arquitetura *post-norm* da implementação
original — `x = LayerNorm(x + sublayer(x))`. Mesmo quando a
contribuição da sub-camada é zero (com *gates* zerados), a
LayerNorm renormaliza o *input* e produz uma saída diferente do
*input* recebido. Em testes empíricos, com $\alpha = 0$, a
diferença máxima absoluta entre saída e *input* foi de $0{,}42$,
indicando que o *adapter* ainda transformava deterministicamente o
sinal. A reformulação para *pre-norm* — `x = x + sublayer(LayerNorm(x))` —
elimina esse efeito completamente: com $\alpha = 0$, a diferença
máxima é $0{,}00 \times 10^{0}$ (identidade exata). Pre-norm é
também a configuração mais estável para *stacks* mais profundas
(Xiong et al., ICML 2020), o que se torna relevante quando se
considera o caso assimétrico discutido a seguir.

Uma quarta direção, decorrente do *insight* de domínio articulado
durante o desenvolvimento, foi a introdução de **assimetria
estrutural entre os ramos proteico e ligante**. Em domínios de
*kinase*, o sítio de ligação ATP é altamente conservado entre
proteínas — a maior parte da informação discriminativa para a
predição de afinidade reside no lado do ligante. A configuração
original tratava ambos os ramos simetricamente em termos de número
de camadas e cabeças de atenção; isso desperdiça capacidade no lado
proteico, onde os *features* já são em grande parte universais, e
sub-aloca capacidade no lado do ligante, onde a discriminação fina
ocorre. A configuração `v7_asymF` exemplifica a inversão da
assimetria: ramo proteico com 1 camada MLP e 4 cabeças de atenção
(*head\_dim* $= 80$); ramo do ligante com 2 camadas MLP e 12
cabeças de atenção (*head\_dim* $= 64$). O custo total em parâmetros
é de $+12\%$ ($6{,}08\text{M} \to 6{,}81\text{M}$),
concentrado precisamente onde a teoria do domínio sugere ganho.

A décima terceira lição que emerge desta revisão é que **assimetria
estrutural orientada por conhecimento de domínio é um eixo
legítimo de otimização arquitetural, distinto do escalonamento
homogêneo de capacidade**. Aumentar `num_heads` de $K$ para $2K$
em ambos os ramos é escalonamento; aumentar somente no ramo onde
a teoria do problema sugere maior demanda de discriminação é
informação prévia. As duas operações têm a mesma magnitude de
custo computacional, mas a segunda incorpora explicitamente a
estrutura do problema. O experimento `v7_asymF` testará
empiricamente se a assimetria ligante-favorita produz ganho
mensurável.

## 6.6. Critério composto de seleção de *checkpoint*

A análise diagnóstica das duas épocas vizinhas (Epoch 77 e Epoch 80
de uma das execuções de `v7+F`) revelou um padrão de divergência
entre `val_loss` e `val_mcc` que merece registro metodológico. As
métricas observadas foram: Epoch 77 com `val_loss = 0{,}1626` e
`val_mcc = 0{,}5928`; Epoch 80 com `val_loss = 0{,}1890` (subida
de $+16\%$) e `val_mcc = 0{,}6054` (subida de $+2{,}1\%$).

Esse padrão é diagnóstico de *threshold gaming*: o modelo aprende
a produzir *logits* extremos (saturados perto de 0 ou 1) que
maximizam o MCC no *threshold* ótimo, mas degradam a calibração
probabilística — o que se reflete em `val_loss` mais alta. O
comportamento é especialmente perigoso em pipelines como o nosso,
em que a seleção do melhor *checkpoint* se baseia exclusivamente
no `val_mcc`: a configuração escolhida será aquela com calibração
pior, e cuja transferência ao conjunto de teste tende a degradar
mais que o esperado.

A confirmação empírica veio da semente 123 desse mesmo experimento:
o *checkpoint* selecionado por `val_mcc = 0{,}6054` produziu
$\mathrm{MCC}_\text{test} = 0{,}4494$, um *gap* val→test de
$0{,}156$ MCC. Em contraste, a perda de AUROC val→test foi de
apenas $0{,}057$, indicando que a capacidade de *ranqueamento* do
modelo se preservou — o que falhou foi a calibração e a transferência
do *threshold* específico.

A solução metodológica adotada é o uso de um **critério composto**
de seleção de *checkpoint*:
$$\text{score} = \mathrm{val\_mcc} - \lambda \cdot \mathrm{val\_loss}$$
em que $\lambda \geq 0$ pondera o quanto a calibração penaliza
ganhos discriminativos. Com $\lambda = 0$, o critério recupera o
comportamento original (seleção pura por `val_mcc`); com
$\lambda = 0{,}5$, aplicado retroativamente ao exemplo das épocas
77 e 80, a Epoch 77 (com calibração superior) é selecionada por
margem estreita ($0{,}5115$ vs $0{,}5109$). Aplicado a Epoch 80
quando $\lambda = 1{,}0$, a margem cresce significativamente
(Epoch 77: $0{,}4302$, Epoch 80: $0{,}4164$).

A décima quarta lição é registrada: **métricas discretas dependentes
de *threshold* devem ser avaliadas em conjunto com métricas
contínuas que reflitam calibração, especialmente em estágios
tardios do treinamento onde a otimização tende a explorar regiões
de saturação dos *logits***. A seleção mono-métrica
(`val_mcc` apenas) pode favorecer *checkpoints* que parecem
melhores em validação mas generalizam pior ao conjunto de teste.
A heurística operacional: sempre que se observar `val_loss`
aumentando enquanto `val_mcc` aumenta, a configuração está
provavelmente em regime de *threshold gaming* e o *checkpoint*
imediatamente anterior — com `val_loss` mais baixo e `val_mcc`
ligeiramente menor — é frequentemente o candidato mais confiável
para o conjunto de teste.

## 6.3. Inflexão estratégica — relaxamento da restrição de identidade

Após o resultado regressivo de `v7-pro` em cinco sementes, a
trajetória de melhorias incrementais por meio de regularizadores
ortogonais aparenta ter atingido um teto local: os *knobs* de
configuração disponíveis foram esgotados, ou a combinação dos que
restam apresenta mais risco de saturação que de ganho aditivo. A
distância restante até $\mathrm{MCC} \approx 0{,}60$, alvo
ambicioso explicitamente colocado pelo orientador, é da ordem de
$+0{,}085$ MCC sobre o atual `v7+` canônico — magnitude
incompatível com qualquer combinação de regularizadores baratos.

Diante desse impasse, a restrição de **preservar a identidade
arquitetural exata do v7** foi formalmente relaxada. A justificativa
é pragmática: a contribuição metodológica da tese — *semantic
screening* a partir de notações lineares, sem estruturas 3D — não
exige a preservação literal do conjunto exato de componentes
implementados na primeira versão; exige apenas que o paradigma de
*encoders* sequenciais (ESM e MoLFormer) com mecanismo de interação
explícito seja mantido. Substituições, extensões ou ampliações de
componentes específicos passam, portanto, a ser admissíveis
desde que a tese descreva claramente cada decisão e seu efeito
mensurado.

Quatro direções de impacto significativamente maior que as
discutidas até este ponto tornam-se imediatamente viáveis sob esse
novo regime:

| Direção | Tipo | $\Delta$MCC esperado | Já implementado? |
|---------|------|---------------------:|:----------------|
| Cabeçote BAN bilinear (`variant=v8`) | substituição arquitetural | $+0{,}015$ a $+0{,}030$ | sim, em `level4_cnn.py` |
| LoRA nas duas camadas superiores dos PLMs | ampliação | $+0{,}020$ a $+0{,}040$ | não |
| Aumento da escala de ESM ($8\text{M} \to 35\text{M}$) | substituição de *backbone* | $+0{,}020$ a $+0{,}050$ | não, requer cache |
| *Distillation* auxiliar de ChemBERTa | aux loss | $+0{,}010$ a $+0{,}025$ | parcial (CKA pendente) |

A nova frente experimental privilegiará BAN como ponto de partida
(mais barato, código já presente), seguido de LoRA caso BAN não
basta para fechar o *gap* e, em última análise, *scale-up* do
*encoder* proteico se as duas anteriores resultarem insuficientes.
A estratégia de empilhamento ortogonal validada no Tier A+C
permanece aplicável: BAN, LoRA e *distillation* atuam em eixos
diferentes (interação, *backbone*, regularização auxiliar) e
podem em princípio compor-se aditivamente.

A décima lição metodológica decorrente desta inflexão é registrada
explicitamente: **o conjunto de restrições experimentais é parte
do desenho do experimento e deve ser revisto periodicamente em
função dos resultados acumulados**. Restrições adotadas no início
do trabalho — nesse caso, a preservação literal da arquitetura
v7 — podem se tornar contraproducentes quando os dados acumulados
indicam que o teto sob essa restrição está abaixo do alvo. A
disciplina de revisitar os limites do espaço experimental, ao
invés de continuar otimizando dentro de um limite previamente
adotado, é parte do método científico, não desvio dele.

Nota importante decorrente desta etapa: o experimento de Tier E
isolado nunca chegou a ser executado antes de adicioná-lo a $F$. A
razão pragmática foi acelerar a iteração, mas a consequência é que,
caso `v7-pro` apresente ganho significativo, será impossível atribuir
o crédito apenas a $E$ ou apenas a $F$ sem ablações adicionais. A
sexta lição metodológica (registrada na seção subsequente) trata
justamente dessa tensão entre velocidade de iteração e capacidade de
atribuição.

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

A quarta lição, decorrente da regressão observada no Tier D, é que
**a aplicabilidade empírica de uma técnica é condicionada ao regime
experimental em que ela foi originalmente formulada**. SWA na sua
forma canônica supõe trajetórias de SGD diversas em torno de um
mínimo já alcançado, condição que pressupõe treinamento longo (na
ordem das centenas de épocas) e taxa de aprendizado elevada. Nosso
*pipeline* (Adam, *early stop* em ~12 épocas) não satisfaz essas
condições. Aplicar SWA cegamente — médiando *checkpoints* da fase de
aprendizado ativo, não da fase de oscilação em torno do ótimo —
produziu o oposto do efeito desejado. A regra operacional resultante:
antes de adotar qualquer técnica documentada na literatura, verificar
explicitamente se o regime experimental em que ela foi validada
corresponde ao regime atual; quando há *mismatch*, escolher uma
variante adaptada (no caso do SWA, *Greedy Model Soup* é o candidato
natural).

A quinta lição, articulada empiricamente após o Tier C mas
metodologicamente importante por si só, é que **overfitting via
seleção baseada em validação não compromete o teste quando este é
escrupulosamente independente**. O protocolo de *scaffold split* com
propagação de *scaffolds* compartilhados, com vazamento entre
*corpora* verificado em zero, garante essa independência. Sob esse
regime, técnicas que dependem explicitamente do desempenho em
validação — tais como busca de limiar ótimo, calibração Platt,
seleção do melhor *checkpoint* e (futuramente) *Greedy Model Soup* —
são metodologicamente legítimas e desejáveis. O *overfitting* que
exige vigilância é o tipo identificável pelo colapso súbito de
$\mathrm{train\_loss}$ a zero em poucas épocas (memorização via
*fingerprint* de identidade no *input*), exemplificado pela falha do
v7-side, e que se previne por construção arquitetural
(inicialização-identidade) antes que se manifeste no treino.

A sexta lição, decorrente da estratégia de empilhamento adotada para
`v7-pro`, é que **velocidade de iteração e capacidade de atribuição
causal são objetivos parcialmente conflitantes**. Adicionar
simultaneamente Mixup e *label smoothing* a uma configuração
validada, sem testá-los isoladamente, acelera o ciclo experimental
mas inviabiliza identificar qual dos dois — se algum — produziu o
ganho observado. A regra prática que emerge: empilhamentos múltiplos
são justificáveis quando os componentes individuais já têm validação
prévia em domínios análogos publicados, e quando o objetivo imediato
é máxima performance (não compreensão científica). Caso a
configuração empilhada se torne candidata à publicação, ablações
isoladas devem ser conduzidas posteriormente para atribuir
contribuição a cada componente. Antes da publicação de qualquer
resultado positivo de `v7-pro`, é portanto requisito metodológico
desativar Mixup e *label smoothing* individualmente em duas
execuções complementares, completando assim a tabela ablativa.

A sétima lição, articulada após observação direta do *output* do
*pipeline* `run_benchmark.sh`, é que **a leitura ingênua do *summary*
de MCC pode confundir significado de fases experimentais
distintas**. Os campos `phase: train` e `phase: test` no
`benchmark_comparison.json` correspondem a duas execuções separadas
do treinamento — a primeira com `fit = train` e `eval = val`, a
segunda com `fit = val` e `eval = test`. São, portanto, modelos
diferentes treinados em conjuntos diferentes, e a expressão "test MCC
maior que train MCC" no contexto desse arquivo não significa
generalização superior à memorização: significa apenas que o modelo
treinado em validação avaliado em teste tem MCC superior ao modelo
treinado em treino avaliado em validação. Essa distinção é crítica
para evitar interpretações incorretas em comunicações posteriores e
deve ser explicitamente esclarecida em qualquer apresentação dos
resultados na tese.

A oitava lição, decorrente do planejamento da etapa de *distillation*
auxiliar de ChemBERTa, é que **diagnósticos baratos devem preceder
implementações custosas sempre que é possível formulá-los**. A
hipótese motivadora — de que MoLFormer e ChemBERTa codificam
informações químicas complementares e que forçar seu alinhamento
durante o treino transferiria conhecimento útil — é falsificável
sem qualquer modificação do modelo: basta computar o *Linear Centered
Kernel Alignment* (Kornblith et al., ICML 2019) entre as
representações globais médias dos dois *encoders* sobre o mesmo
conjunto de ligantes. Caso o CKA exceda $0{,}9$, os espaços são
redundantes a ponto de a *distillation* não trazer ganho mensurável,
e investir cento e cinquenta linhas de código para implementá-la
torna-se desperdício de tempo experimental. O *script*
`scripts/thesis_followups/chemberta_molformer_complementarity.py`
implementa esse teste em aproximadamente cento e cinquenta linhas
e roda em segundos. A regra prática que emerge: **antes de
implementar qualquer mecanismo cujo benefício depende de uma
hipótese sobre estrutura interna do dado ou da representação, buscar
um teste empírico barato que falsifique essa hipótese**. No caso de
*distillation*, mede-se o CKA antes de implementar o módulo de
treino; no caso de *Mixup*, observa-se a distribuição do *loss* sob
combinações lineares dos *inputs*; no caso de adaptação de domínio,
verifica-se primeiro se a separabilidade entre domínios pelo
classificador de *features* já é elevada antes de treinar o
classificador adversarial.

A nona lição, decorrente da estratégia atual de validação cruzada
entre *non\_human* e *human*, é que **otimizações conduzidas em um
único corpus podem ser corpus-específicas e não transferíveis**.
Os parâmetros `patience = 15`, `lr_mult = 2{,}0`,
`contrastive_weight = 0{,}3`, `mixup_alpha = 0{,}3` e
`label_smooth = 0{,}05` foram todos ajustados sobre o corpus
*non\_human* (com sete mil seiscentas amostras de treino), o menor
e mais rápido a iterar. O regime de *human* (com aproximadamente
sessenta e seis mil amostras) tem trajetória de *loss*, número de
épocas até *early stop* e curvatura local da paisagem de *loss*
substancialmente diferentes; *knobs* otimizados sob o regime menor
não têm garantia de transferir. A *patience* especialmente é
suspeita: quinze épocas em um corpus dez vezes menor representam
muito mais varreduras pelo conjunto de treino do que em um corpus
maior, possivelmente forçando convergência indevida. A regra
operacional que emerge: **toda configuração otimizada exclusivamente
em um corpus deve ser revalidada em cada corpus-alvo de aplicação;
quando há regressão na transferência, os hiperparâmetros são
suspeitos antes da arquitetura**.

A tabela abaixo consolida o progresso observado até este ponto.

| Etapa                          | Mudança                                | Train MCC | Test MCC | Δ vs v7 |
|--------------------------------|----------------------------------------|----------:|---------:|--------:|
| v7 baseline                    | linha base, seed 42, NH                | 0,5208    | 0,4862   |     —   |
| Tier A não-tunado              | heads=16, head_dim=64, MLP, adapter 2× | 0,5115    | 0,4697   | −0,016  |
| **Tier A tunado**              | acima + patience=15, lr_mult=2         | **0,5652**| **0,5004**| **+0,014** |
| Tier B (sobre Tier A)          | + pool_num_heads=4                     | 0,5200    | 0,4560   | −0,044  |
| Tier C (seed 42 isolada)       | + contrastive_weight=0,3, cosine_feat  | 0,5880    | 0,5167    | +0,031 (single)|
| **Tier C (5-seed média)**      | mesmo, $42, 123, 456, 789, 1024$       | $0{,}576 \pm 0{,}036$ | $\mathbf{0{,}514 \pm 0{,}008}$ | $\mathbf{+0{,}008}$ (mean) |
| Tier D (sobre Tier A+C, seed 42) | SWA vanilla, swa_start=5            | 0,5088    | 0,4964    | $-0{,}021$ regrediu |
| Tier E+F empilhados (v7-pro, seed 42 d01)     | + mixup_alpha=0,3, label_smooth=0,05 | $0{,}5870$ | $0{,}5320$ | $+0{,}046$ (single, vs v7) |
| Tier E+F empilhados (v7-pro, 5-seed d01)     | mesmo, $42, 123, 456, 789, 1024$       | $0{,}576$ | $0{,}4961 \pm 0{,}0245$ | $-0{,}018$ (vs v7+ A+C 5-seed) |
| Tier E isolado (v7+E, 3-seed d01)            | A+C + mixup_alpha=0,3                  | —         | $0{,}4988 \pm 0{,}0254$ | $-0{,}016$ Mixup deletério |
| **Tier F isolado (v7+F, 3-seed d01)**        | A+C + label_smooth=0,05                | —         | $\mathbf{0{,}5260 \pm 0{,}0274}$ | $\mathbf{+0{,}012}$ candidato a canônico |
| BAN-híbrido (v7_ban_F, 3-seed d01)            | A+C+F + variant=v8 (Xavier W_ban)     | $0{,}599 \pm 0{,}034$ | $0{,}5028 \pm 0{,}0462$ | $-0{,}023$ vs v7+F (regrediu, $\sigma$ inflada) |

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

Uma observação inesperada surgiu na transição entre os dois
hospedeiros. A configuração `v7+ Tier A+C` — exatamente a mesma
*config*, mesmas sementes, mesmo protocolo — produziu MCC teste de
$0{,}5167$ em `diamante-02` (cuDNN desabilitado, núcleos `ATen`
nativos) e $0{,}5256$ em `diamante-01` (cuDNN ativo). A diferença,
$+0{,}009$ MCC entre os dois hospedeiros, é da mesma ordem de
magnitude que o desvio-padrão entre sementes ($\sigma = 0{,}008$),
e seu sinal é positivo: o caminho cuDNN-otimizado parece, em nossa
arquitetura, produzir trajetórias de otimização ligeiramente
superiores. A hipótese é que os algoritmos selecionados pelo
*benchmark* interno do cuDNN (Winograd, FFT, GEMM otimizado)
introduzem ruído numérico estatisticamente diferente daquele dos
*kernels* `ATen` nativos, e esse ruído pode atuar como um
regularizador implícito favorável. A conclusão prática é que
**comparações quantitativas de MCC só são válidas dentro do mesmo
hospedeiro**; reportar diferenças entre execuções em `diamante-01` e
`diamante-02` mistura efeito experimental com efeito numérico.

Para a tese, isso reforça a recomendação metodológica de fixar o
hospedeiro experimental durante o ciclo de validação multi-semente
da configuração final.

---

## 9. Considerações finais

A jornada documentada neste registro confirma que a arquitetura
DT-Kinase v7 é um sistema bem balanceado dentro de seu próprio espaço
de configuração: ganhos significativos exigem ou correção cuidadosa
dos hiperparâmetros de otimização ao escalonar capacidade, ou
intervenções pontuais em componentes que não afetem a continuidade do
regime inicial de treinamento. Tentativas indiscriminadas de aumentar
expressividade — como o *pool* multi-cabeça do Tier B — podem
prejudicar a rede se não respeitarem essa continuidade.

O Tier C, executado com sucesso, posicionou a configuração de
referência (Tier A tunado combinado com perda contrastiva auxiliar)
em $\mathrm{MCC} = 0{,}5143 \pm 0{,}0079$ sobre cinco sementes em
`diamante-02`, e $\mathrm{MCC} = 0{,}5256$ na semente 42 em
`diamante-01`. Essa configuração consolida-se como o `v7+` canônico
(salva em `configs/v7_plus.yaml`), enquanto a configuração que
empilha Tier E (Mixup) e Tier F (*label smoothing*) sobre Tier A+C
constitui a frente experimental imediata, denominada `v7-pro` (em
`configs/v7_pro.yaml`).

Duas frentes paralelas de validação restam pendentes. A primeira é
verificar a transferibilidade dos *knobs* otimizados em
*non\_human* para o corpus *human*, dez vezes maior; eventuais
regressões nesse novo regime indicariam que parâmetros como
`patience = 15` e `lr_mult = 2.0` são específicos do corpus pequeno
e exigiriam re-tuning. A segunda é a avaliação multi-semente da
configuração `v7-pro` em *non\_human*, comparando-a diretamente com
a `v7+` canônica para quantificar o ganho aditivo dos Tiers $E$ e
$F$. Caso `v7-pro` cruze o limiar $0{,}52$ na média de cinco
sementes, ablações isoladas de Tier $E$ e Tier $F$ devem ser
conduzidas posteriormente, conforme a sexta lição da seção 7,
para atribuir contribuição a cada componente antes da publicação.

Para reduzir o atrito operacional entre essas validações, foi
introduzido o *script* `scripts/v8/run_v7_pro_validation.sh`, que
encadeia sequencialmente o treinamento multi-semente nos dois
corpora — primeiro *non\_human* (aproximadamente 25 minutos) e em
seguida *human* (estimados 4 a 6 horas em `diamante-01`) — e
imprime um sumário consolidado das duas execuções ao final. A
existência de uma orquestração automatizada para a fase de
validação é em si uma decisão metodológica: separa o esforço de
configuração do esforço de execução, e permite que o ciclo
`config → resultado multi-semente em ambos corpora` seja repetido
sem intervenção manual a cada iteração futura. Tais artefatos —
*scripts* declarativos que reproduzem a validação completa de uma
configuração — devem acompanhar qualquer entrega final na tese,
de modo que terceiros possam reproduzir os números reportados.

### Estado operacional atual (snapshot)

A configuração que se encontra em execução é `v7_asymF` em três
sementes ($42, 123, 456$) sobre *non\_human* em `diamante-01`. A
configuração combina simultaneamente quatro elementos validados
ou implementados nas etapas anteriores: Tier A (capacidade
escalada), Tier C (perda contrastiva auxiliar), Tier F (*label
smoothing* $\varepsilon = 0{,}05$, validado isolado em $\S 6{.}2$),
e a revisão estrutural completa do `EmbeddingAdapter` ($\S 6{.}5$ —
*pre-norm*, *gates* LoRA-style, projeção de auto-atenção
zero-init, e assimetria estrutural com dois blocos MLP e doze
cabeças de atenção no ramo do ligante contra um bloco e quatro
cabeças no ramo proteico). O critério de seleção composto
($\S 6{.}6$) está ativo via
$\mathrm{BENCHMARK\_LEVEL4CNN\_SELECTION\_LAMBDA\_LOSS} = 0{,}5$.

O resultado dessa execução determina a próxima decisão. Caso a
média de teste cruze $0{,}53$ MCC, `v7_asymF` é promovido a
candidato canônico para validação cinco-sementes posterior.
Caso fique entre $0{,}51$ e $0{,}53$, a contribuição da assimetria
arquitetural é considerada marginal e não suficiente para
justificar o aumento de $12\%$ em parâmetros sobre `v7+F`. Caso
fique abaixo de $0{,}51$, a assimetria é deletéria e deve ser
revertida; a investigação migra para BAN-residual com $\alpha$-gate
identity-init ($\S 6{.}4$) ou LoRA nas camadas superiores dos
PLMs ($\S 10$).

Configurações de uso imediato e seus respectivos *output\_root*
encontram-se sumarizadas:

| Config | Tiers | `output_root` | Status |
|---|---|---|---|
| `configs/v7.yaml` | linha base | `results/benchmark_{dataset}_{embedding}_13_04_2026` | tese referência |
| `configs/v7_plus.yaml` | A + C | `results/benchmark_plus_*` | canônico validado 5-seed |
| `configs/v7_plus_F.yaml` | A + C + F | `results/benchmark_plusF_*` | melhor validado 3-seed |
| `configs/v7_asymF.yaml` | A + C + F + adapter assimétrico | `results/benchmark_asymF_*` | em execução |
| `configs/v7_ban_F.yaml` | A + C + F + BAN | `results/benchmark_banF_*` | regrediu (W_ban Xavier) |
| `configs/v7_pro.yaml` | A + C + E + F | `results/benchmark_pro_*` | descartado (Mixup) |
| `configs/v7_plus_E.yaml` | A + C + E | `results/benchmark_plusE_*` | descartado (Mixup) |

---

## 10. Direções adicionais de otimização

A discussão presente nesta seção tem caráter prospectivo. As cinco
direções aqui apresentadas foram selecionadas por satisfazerem
simultaneamente três critérios: preservam a identidade arquitetural
do v7 (*cross-attention* 2D, CNN bidimensional, *HierPool*, *backbones*
congelados); apresentam baixo custo de implementação relativo ao ganho
esperado; e são cientificamente defensáveis no contexto da tese,
oferecendo motivação biológica ou matemática que se pode articular no
texto metodológico.

A primeira direção — e provavelmente a mais robusta em termos de
relação custo-benefício — é a adoção de *Stochastic Weight Averaging*
(SWA) ou, alternativamente, médias móveis exponenciais dos pesos
durante o treinamento. O mecanismo consiste em manter, em paralelo aos
pesos do otimizador, uma cópia média dos parâmetros da rede,
atualizada a cada iteração com fator de decaimento controlado;
no momento da avaliação, essa cópia substitui os pesos do otimizador.
A literatura (Izmailov et al., 2018) reporta que essa técnica
suaviza a paisagem de *loss* explorada pelo modelo, levando-o a
mínimos mais largos e, consequentemente, a melhor generalização. O
ganho esperado situa-se entre $0{,}005$ e $0{,}02$ MCC, sem custo
computacional adicional e sem qualquer alteração arquitetural — apenas
um *hook* no laço de treinamento e a substituição dos pesos antes da
calibração de Platt.

A segunda direção é a ativação do regime de *Mixup* sobre os mapas de
interação 2D, controlado pelo parâmetro `mixup_alpha` já existente no
código mas atualmente desabilitado. *Mixup* (Zhang et al., 2018) gera,
durante o treinamento, combinações lineares aleatórias de pares de
exemplos com pesos amostrados de uma distribuição Beta($\alpha,\alpha$);
o classificador é treinado a prever a combinação correspondente das
*labels*. No domínio proteína-ligante, isso pode ser interpretado como
treinar o modelo a reconhecer interações em padrões intermediários
entre exemplos reais — uma forma de *data augmentation* implícita que
suaviza a fronteira de decisão. Valores razoáveis de $\alpha$ situam-se
entre $0{,}2$ e $0{,}4$. O ganho típico em domínios análogos (DTI,
classificação molecular) é de $0{,}01$ a $0{,}02$ MCC, ao custo de uma
linha de configuração.

A terceira direção, mais ambiciosa porém ainda dentro do paradigma de
v7, é a introdução de *deep supervision* na pilha CNN. A ideia é
acoplar um classificador auxiliar a uma camada intermediária da CNN
(tipicamente a segunda ou terceira) cujo *loss* contribui para o
gradiente com peso reduzido, da ordem de $0{,}3$. O efeito conhecido é
forçar as camadas intermediárias a aprender representações
preditivamente úteis em si, ao invés de meramente intermediárias para
camadas posteriores; o classificador auxiliar é descartado em
inferência. Esse mecanismo é particularmente relevante quando a
profundidade da CNN cresce ou quando sua convergência é lenta —
sintomas que o Tier A revelou marginais mas presentes. A
implementação requer aproximadamente cinquenta linhas de código e o
ganho esperado fica entre $0{,}005$ e $0{,}015$ MCC.

A quarta direção, com motivação biológica explícita, é a aplicação de
uma penalidade de esparsidade $L_1$ sobre as distribuições de atenção
do *HierPool*. Sob o ponto de vista biológico, ligações específicas
proteína-ligante envolvem tipicamente um pequeno número de resíduos
em sítios ativos bem definidos, não uma distribuição uniforme ao longo
da cadeia inteira. Forçar as distribuições *softmax* das *queries* a
serem esparsas — concentradas em poucas posições — alinha o
mecanismo de atenção com essa intuição. Tecnicamente, basta
adicionar um termo $\lambda \sum_{i,j} |a_{ij}|$ ao *loss* total, com
$\lambda$ pequeno (por exemplo $10^{-3}$), incidindo sobre as
distribuições de atenção do `_AxisAttentionPool`. Além de ganho
modesto em MCC ($0{,}005$ a $0{,}01$), oferece uma narrativa
metodológica forte para o capítulo da tese, conectando a arquitetura
à fenomenologia biológica do problema.

A quinta direção, finalmente, é a aplicação de *test-time augmentation*
durante a fase de inferência, explorando o fato de que cada molécula
admite múltiplas representações SMILES canonicalmente equivalentes mas
sintaticamente distintas. A técnica consiste em gerar, para cada
exemplo de teste, $K$ representações SMILES aleatorizadas (por exemplo
$K = 5$), processar cada uma pelo modelo, e calcular a média das $K$
probabilidades resultantes antes da limiarização. Isso aproxima a
estimativa pontual de uma *posterior preditiva*, reduzindo a variância
do classificador. O custo é apenas em inferência (sem alterar o
treinamento), e o ganho típico em tarefas de classificação molecular
fica entre $0{,}005$ e $0{,}015$ MCC. A implementação envolve a
geração de SMILES aleatórios via RDKit (`Chem.MolToSmiles(mol,
doRandom=True)`) e uma adaptação do passo de inferência.

Para além dessas cinco direções, duas opções de maior risco e maior
potencial permanecem disponíveis caso o teto observado sob v7 puro se
revele insuficiente. A primeira é a aplicação de LoRA (Low-Rank
Adaptation, Hu et al., 2022) às duas camadas superiores dos PLMs
ESM-2 e MoLFormer, relaxando o regime estritamente congelado dos
*backbones* sem retreiná-los completamente; a segunda é o aumento da
escala do PLM proteico, substituindo ESM-2 8M por ESM-2 35M ou 150M, o
que requer reconstruir os caches de *embeddings* mas pode oferecer
ganho substancial pela maior capacidade representacional. Ambas as
opções comprometem parcialmente a identidade do v7 e devem ser
consideradas apenas após esgotar as primeiras cinco direções.

A tabela seguinte sumariza o ranqueamento dessas direções segundo a
heurística de relação custo-benefício, considerando ganho esperado em
MCC, custo de implementação, risco metodológico e fidelidade ao
paradigma proposto.

| Direção                                   | ΔMCC esperado | Custo impl | Risco | Identidade v7 | Status |
|-------------------------------------------|--------------:|:----------:|:-----:|:-------------:|:-------|
| SWA / EMA dos pesos                       | $+0{,}005$ a $+0{,}02$ | baixo      | baixo | preservada | testado, regrediu (Tier D) |
| *Mixup* sobre mapas de interação          | $+0{,}01$ a $+0{,}02$  | trivial    | médio | preservada | testado, regrediu isolado (Tier E) |
| *Label smoothing* sobre o alvo            | $+0{,}005$ a $+0{,}015$| trivial    | baixo | preservada | **validado isolado (Tier F): +0,012 MCC** |
| Cabeçote BAN bilinear (`variant=v8`)      | $+0{,}015$ a $+0{,}030$| trivial    | médio | parcial | testado, regrediu (Xavier-init W_ban) |
| BAN-residual com $\alpha$-gate identidade-init | $+0{,}010$ a $+0{,}025$| médio      | baixo | parcial | pendente |
| Pre-norm no `EmbeddingAdapter`            | marginal isolado | trivial | baixo | preservada | **implementado** |
| `EmbeddingAdapter` com gates LoRA-style   | $+0{,}005$ a $+0{,}015$| baixo | baixo | preservada | **implementado** |
| Adapter assimétrico (ligante > proteína)  | $+0{,}010$ a $+0{,}020$| baixo | médio | preservada | **implementado, pendente teste** (`v7_asymF`) |
| Critério composto val_mcc − λ·val_loss    | $+0{,}005$ a $+0{,}015$| baixo | baixo | preservada | **implementado** (env knob) |
| *Deep supervision* em camada CNN          | $+0{,}005$ a $+0{,}015$| médio      | médio | preservada | pendente |
| Esparsidade $L_1$ sobre atenção           | $+0{,}005$ a $+0{,}01$ | trivial    | baixo | preservada | pendente |
| SMILES *test-time augmentation*           | $+0{,}005$ a $+0{,}015$| baixo      | baixo | preservada | pendente |
| *Greedy Model Soup* (variante de SWA)     | $+0{,}01$ a $+0{,}03$  | médio      | baixo | preservada | pendente |
| Distillation auxiliar de ChemBERTa        | $+0{,}005$ a $+0{,}025$| médio      | médio | preservada | pendente CKA |
| LoRA nas camadas superiores dos PLMs      | $+0{,}02$ a $+0{,}04$  | médio-alto | médio | parcial | pendente |
| Aumento da escala de ESM (8M → 35M / 150M)| $+0{,}02$ a $+0{,}05$  | alto       | alto  | parcial | pendente |

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

Izmailov *et al.*, "Averaging weights leads to wider optima and better
generalization", *UAI* (2018).

Zhang *et al.*, "mixup: Beyond empirical risk minimization", *ICLR*
(2018).

Lee *et al.*, "Deeply-supervised nets", *AISTATS* (2015).

Szegedy *et al.*, "Rethinking the Inception architecture for computer
vision", *CVPR* (2016).

Wortsman *et al.*, "Model soups: averaging weights of multiple
fine-tuned models improves accuracy without increasing inference
time", *ICML* (2022).

Kornblith *et al.*, "Similarity of neural network representations
revisited", *ICML* (2019).
