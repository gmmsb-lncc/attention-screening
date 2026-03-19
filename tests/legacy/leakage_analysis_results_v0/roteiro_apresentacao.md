# Roteiro da Apresentação: Análise de Vazamento de Dados

## Guia Completo para Apresentadores e Leigos

Este roteiro explica cada slide da apresentação de forma didática, com definições de termos técnicos e analogias para facilitar a compreensão.

---

## Glossário de Termos Essenciais

Antes de começar, vamos definir alguns termos que aparecerão frequentemente:

| Termo | Definição Simples |
|-------|-------------------|
| **Quinase** | Uma proteína do corpo humano que funciona como um "interruptor" celular. Quando está defeituosa, pode causar doenças como câncer. |
| **Composto químico** | Uma molécula criada em laboratório que pode virar um medicamento. |
| **Ativo** | Um composto que consegue "desligar" ou "modificar" a quinase (potencial medicamento). |
| **Inativo** | Um composto que não tem efeito sobre a quinase. |
| **Dataset** | Conjunto de dados usado para treinar o modelo. |
| **Treino/Teste** | Dividimos os dados em duas partes: uma para "ensinar" o modelo (treino) e outra para verificar se ele aprendeu de verdade (teste). |
| **Accuracy (Acurácia)** | Porcentagem de acertos do modelo. Se acerta 87 de 100, tem 87% de acurácia. |
| **MCC** | Matthews Correlation Coefficient - uma métrica mais rigorosa que a acurácia, especialmente útil quando há desbalanceamento de classes. Vai de -1 (péssimo) a +1 (perfeito), onde 0 é aleatório. |
| **Fingerprint molecular** | Uma "impressão digital" do composto - um código numérico que representa sua estrutura química. |
| **One-hot encoding** | Uma forma de representar categorias como números. Ex: se temos 3 quinases (A, B, C), a quinase B seria representada como [0, 1, 0]. |

---

## SLIDE 1: Capa

**O que dizer:**
> "Boa tarde a todos. Hoje vou apresentar uma análise crítica sobre por que modelos de machine learning aparentemente simples conseguem resultados extraordinários na predição de atividade entre compostos químicos e quinases. Spoiler: a resposta não é que os modelos são geniais - é que há problemas sérios nos dados."

---

## SLIDE 2: Sumário

**O que dizer:**
> "Vamos passar por 8 tópicos principais. Começaremos entendendo o contexto do problema, depois vamos investigar diversos tipos de 'vazamentos' e vieses nos dados, e finalmente veremos o impacto real disso na performance dos modelos."

---

## SLIDE 3: Contexto - Predição de Atividade Quinase-Composto

**O que dizer:**
> "Primeiro, vamos entender o problema que estamos tentando resolver. Quinases são proteínas muito importantes no nosso corpo - elas funcionam como 'interruptores' que controlam processos celulares. Quando uma quinase está defeituosa, pode causar doenças como câncer.

> O objetivo é descobrir quais compostos químicos conseguem 'desligar' ou modificar essas quinases defeituosas. Se conseguirmos, temos um potencial medicamento.

> Na prática, temos um problema de classificação binária: dado um composto e uma quinase, queremos predizer se o composto é ATIVO (funciona contra a quinase) ou INATIVO (não funciona).

> Usamos representações simples:
> - Para quinases: one-hot encoding (basicamente um código de identificação)
> - Para compostos: Morgan Fingerprints, que são como uma 'impressão digital' da molécula com 2048 números

> Nosso dataset tem cerca de 15 mil pares composto-quinase, com 8 mil compostos diferentes testados contra 231 quinases diferentes."

**Analogia para leigos:**
> "Imagine que você tem 8 mil chaves diferentes (compostos) e 231 fechaduras (quinases). Queremos descobrir quais chaves abrem quais fechaduras, sem ter que testar todas as combinações no laboratório."

---

## SLIDE 4: O Problema - Performance Inesperadamente Alta

**O que dizer:**
> "Aqui está o mistério que nos motivou a fazer essa análise. Quando treinamos modelos simples como KNN e MLP com essas representações básicas, obtivemos resultados surpreendentemente bons:
> - KNN: 87.6% de acurácia e 0.747 de MCC
> - MLP: 88.7% de acurácia e 0.768 de MCC

> Isso parece ótimo, certo? Mas há um problema: esse é um problema de biologia molecular extremamente complexo! Como é possível que representações tão simples resolvam um problema tão difícil?

> Nossa hipótese: os modelos não estão realmente 'aprendendo' padrões biológicos - eles estão 'trapaceando' de alguma forma."

**Analogia para leigos:**
> "É como se um aluno tirasse 90% na prova sem estudar. Você desconfiaria que ele colou, certo? É exatamente isso que estamos investigando."

---

## SLIDE 5: Problema Crítico - Vazamento de Compostos

**O que dizer:**
> "E encontramos a primeira 'cola': vazamento de dados!

> Quando dividimos os dados em treino e teste, o ideal é que o teste contenha situações NOVAS que o modelo nunca viu. Mas descobrimos que:
> - 64,7% das linhas de teste contêm compostos que JÁ APARECERAM no treino
> - 36,6% são duplicatas EXATAS - o mesmo composto testado contra a mesma quinase!

> Para ser mais preciso: dos 1.355 compostos únicos no teste, 820 deles (60,5%) também estão no treino. E como alguns compostos aparecem várias vezes, isso representa 64,7% das linhas."

**Analogia para leigos:**
> "Imagine uma prova de matemática onde 65% das questões são IDÊNTICAS às que o aluno já resolveu no dever de casa. Claro que ele vai tirar nota alta! Mas isso não significa que ele sabe matemática - significa que ele tem boa memória.

> O mesmo acontece aqui: o modelo não está aprendendo a prever se um composto novo vai funcionar - ele está apenas 'lembrando' a resposta de compostos que já viu."

---

## SLIDE 6: Hipótese - Modelo Simples de "Lookup"

**O que dizer:**
> "Para provar que os modelos estão memorizando, criamos modelos extremamente simples chamados 'Lookup' (que significa 'consulta' ou 'olhar na tabela').

> Esses modelos não fazem nenhum cálculo sofisticado. Eles simplesmente:
> 1. **Lookup por Composto**: Olha qual foi a classe mais comum desse composto no treino e repete
> 2. **Lookup por Quinase**: Olha qual foi a classe mais comum dessa quinase no treino e repete
> 3. **Lookup Combinado**: Combina as duas estratégias

> Se o modelo KNN/MLP estivesse realmente aprendendo padrões complexos, ele deveria ser MUITO melhor que esses modelos triviais."

**Analogia para leigos:**
> "É como se, em vez de resolver a conta de matemática, você apenas olhasse: 'Hmm, toda vez que apareceu o número 7 na prova, a resposta foi B. Vou marcar B.' Isso é o Lookup."

---

## SLIDE 7: Resultados - Lookup vs KNN (Gráfico)

**O que dizer:**
> "E aqui está o resultado chocante. Olhem o gráfico:
> - O modelo Lookup Composto+Quinase, que APENAS consulta a classe majoritária, consegue 86,2% de acurácia e 0,717 de MCC
> - O KNN, com todo seu algoritmo de vizinhos mais próximos, consegue 87,6% de acurácia e 0,747 de MCC

> A diferença é mínima! Um modelo que simplesmente 'lembra' a resposta tem quase a mesma performance que o KNN."

---

## SLIDE 8: Análise dos Resultados Lookup

**O que dizer:**
> "Vamos analisar os números com calma:

> O Lookup por Composto sozinho já consegue quase 79% de acurácia. Isso significa que, só de saber qual é o composto, já conseguimos acertar quase 80% das vezes!

> Quando combinamos Composto + Quinase, chegamos a 86,2% - apenas 1,4 pontos percentuais abaixo do KNN.

> A conclusão é clara: o KNN não está aprendendo interações complexas entre compostos e quinases. Ele está, essencialmente, memorizando."

---

## SLIDE 9: Distribuição de Classes por Quinase (Gráfico)

**O que dizer:**
> "Agora vamos entender OUTRO problema: o desbalanceamento por quinase.

> O gráfico da esquerda mostra a distribuição de classes para cada quinase. Cada barra representa quantas quinases têm uma determinada proporção de ativos.

> Vejam os extremos:
> - Muitas quinases têm 0% de ativos (todas as interações são inativas)
> - Outras têm 100% de ativos (todas as interações são ativas)

> O gráfico de pizza à direita resume: 70,6% das quinases são extremamente desbalanceadas!"

---

## SLIDE 10: Quinases Trivialmente Previsíveis

**O que dizer:**
> "O que isso significa na prática? Significa que, para a maioria das quinases, o modelo não precisa aprender nada - basta chutar a classe majoritária!

> Por exemplo:
> - Para a Glycogen synthase kinase-3 beta, 100% dos compostos são ativos. O modelo só precisa aprender: 'se for essa quinase, responda ATIVO'
> - Para muitas outras quinases, 100% são inativos. O modelo só precisa responder INATIVO

> Isso explica por que o Lookup por Quinase sozinho já consegue 75% de acurácia!"

**Analogia para leigos:**
> "Imagine uma prova de múltipla escolha onde, em 70% das questões, a resposta correta é sempre a letra A. Se você marcar A em tudo, já garante 70%!"

---

## SLIDE 11: Comportamento dos Compostos através de Quinases (Gráfico)

**O que dizer:**
> "Agora vamos analisar o comportamento dos COMPOSTOS. O gráfico mostra a 'consistência' de cada composto.

> **O que significa consistência?**
> Um composto é 'consistente' quando ele se comporta da mesma forma contra TODAS as quinases que foi testado.
> - Se um composto é SEMPRE ativo (contra todas as quinases testadas), ele é consistente
> - Se um composto é SEMPRE inativo, também é consistente
> - Se um composto é ativo contra algumas quinases e inativo contra outras, ele é INCONSISTENTE

> O gráfico de pizza mostra:
> - 80,2% dos compostos foram testados contra apenas UMA quinase
> - 15,2% são perfeitamente consistentes (sempre ativo OU sempre inativo)
> - Apenas 4,6% são inconsistentes"

**Analogia para leigos:**
> "Pense em medicamentos do dia a dia:
> - O paracetamol funciona para dor de cabeça, dor muscular, febre... É 'consistente' - funciona para várias coisas
> - Alguns medicamentos são muito específicos - só funcionam para uma doença
> - Raramente um medicamento funciona para gripe mas piora dor de cabeça - isso seria 'inconsistente'

> Nossos dados mostram que a maioria dos compostos é consistente: se funciona para uma quinase, provavelmente funciona para outras."

---

## SLIDE 12: Análise de Consistência

**O que dizer:**
> "Vamos aos números dos compostos testados contra MÚLTIPLAS quinases:
> 76,7% deles são perfeitamente consistentes!

> **O que isso implica?**
> Se um composto se comporta da mesma forma contra todas as quinases, então a informação mais importante está no COMPOSTO, não na quinase específica.

> Ou seja: só de olhar o fingerprint do composto, já conseguimos prever se ele será ativo ou inativo - independente de qual quinase estamos testando!

> Isso é problemático porque significa que o modelo não está aprendendo INTERAÇÕES específicas entre composto e quinase. Ele está apenas categorizando compostos como 'bons' ou 'ruins'."

---

## SLIDE 13: Similaridade Tanimoto - Teste vs Treino (Gráfico)

**O que dizer:**
> "Mesmo para os compostos 'novos' no teste (aqueles que não aparecem no treino), existe outro problema: eles são MUITO SIMILARES aos compostos do treino.

> O gráfico mostra a distribuição de similaridade de Tanimoto, que mede quão parecidos dois compostos são quimicamente (0 = totalmente diferentes, 1 = idênticos).

> Para cada composto novo no teste, calculamos: qual o composto mais parecido no treino?

> Resultados:
> - 43% têm similaridade > 0.8 (muito similares)
> - 46,8% têm similaridade entre 0.6 e 0.8 (similares)
> - Apenas 4,4% têm similaridade < 0.4 (realmente diferentes)"

**Analogia para leigos:**
> "Imagine que você treina um modelo para reconhecer fotos de gatos. No teste, você coloca fotos dos MESMOS gatos, só que de ângulos ligeiramente diferentes. Claro que o modelo vai acertar! Mas ele realmente aprendeu a reconhecer gatos em geral?"

---

## SLIDE 14: Por que o KNN Funciona?

**O que dizer:**
> "Agora entendemos por que o KNN (K-Nearest Neighbors, ou K-Vizinhos Mais Próximos) funciona tão bem!

> O KNN funciona assim: quando recebe um composto novo, ele procura os compostos mais PARECIDOS no treino e copia a resposta deles.

> Se 90% dos compostos de teste têm um 'vizinho' muito similar no treino, então o KNN quase sempre vai encontrar a resposta correta - não porque entendeu a biologia, mas porque o composto de teste é quase idêntico a algo que ele já viu.

> É como ter uma prova onde 90% das questões são reformulações de exercícios do livro. Se você decorou o livro, vai bem - mas não significa que entendeu a matéria."

---

## SLIDE 15: Três Cenários de Split

**O que dizer:**
> "Para provar definitivamente nosso ponto, criamos três cenários de avaliação:

> **Cenário 1 - Split Aleatório (Original):**
> Dividimos os dados aleatoriamente - é o que foi feito originalmente. Tem todos os problemas de vazamento que discutimos.

> **Cenário 2 - Split por Composto:**
> Garantimos que NENHUM composto do teste aparece no treino. Todos os compostos no teste são 'novos'.

> **Cenário 3 - Novo Composto + Nova Quinase:**
> O cenário mais rigoroso: os compostos E as quinases do teste NUNCA foram vistos no treino. Isso simula o caso de uso REAL - quando queremos prever atividade para um medicamento totalmente novo contra uma quinase nunca estudada."

---

## SLIDE 16: Resultados por Cenário de Avaliação (Gráfico)

**O que dizer:**
> "E aqui está a prova definitiva. Olhem como a performance DESPENCA quando eliminamos o vazamento:

> No split aleatório, tínhamos ~87% de acurácia e ~0.75 de MCC.

> No split de generalização verdadeira (Cenário 3):
> - KNN cai para 68% de acurácia e 0.41 de MCC
> - MLP cai para 60% de acurácia e 0.27 de MCC

> Notem também os tamanhos dos conjuntos de teste abaixo de cada barra - no Cenário 3 temos menos dados porque é mais restritivo."

---

## SLIDE 17: Performance Inflada vs Performance Real (Gráfico)

**O que dizer:**
> "Este gráfico resume tudo. À esquerda, o MCC; à direita, a acurácia.

> As barras azuis mostram a performance 'reportada' (com vazamento).
> As barras vermelhas mostram a performance 'real' (generalização verdadeira).

> Vejam as setas com as porcentagens de queda:
> - MCC do KNN cai 45%
> - MCC do MLP cai 64%!

> O MLP, que parecia ser o melhor modelo, sofre a maior queda! Isso sugere que ele era o mais dependente do vazamento."

---

## SLIDE 18: Queda Dramática de Performance

**O que dizer:**
> "Vamos aos números exatos:

> **MCC (métrica mais importante):**
> - KNN: 0.747 → 0.407 (queda de 45%)
> - MLP: 0.768 → 0.275 (queda de 64%)

> **Acurácia:**
> - KNN: 87.6% → 68.0% (queda de 22%)
> - MLP: 88.7% → 60.7% (queda de 32%)

> Um MCC de 0.275 é muito próximo de aleatório! Significa que o MLP, no cenário real, praticamente não consegue distinguir ativos de inativos.

> Isso confirma nossa hipótese: os modelos estavam MEMORIZANDO, não aprendendo."

---

## SLIDE 19: Resumo das Descobertas

**O que dizer:**
> "Vamos resumir as cinco descobertas principais:

> 1. **Vazamento Massivo**: 60% dos compostos de teste estão no treino
> 2. **Desbalanceamento Extremo**: 70% das quinases são trivialmente previsíveis
> 3. **Consistência de Compostos**: O fingerprint sozinho carrega o sinal preditivo
> 4. **Alta Similaridade**: 90% dos compostos de teste são muito similares ao treino
> 5. **Performance Inflada**: MCC cai de 0.77 para 0.27 em cenário real

> Cada um desses problemas contribui para inflar artificialmente a performance."

---

## SLIDE 20: Implicações para a Área

**O que dizer:**
> "E aqui está a crítica central deste trabalho:

> **Modelos de triagem de quinases publicados na literatura, que não fizeram essa curadoria cuidadosa dos dados, podem estar DRASTICAMENTE superestimando sua capacidade de generalização!**

> Isso é grave porque esses modelos são usados para:
> - Priorizar compostos para testes em laboratório
> - Economizar tempo e dinheiro em pesquisa farmacêutica
> - Tomar decisões sobre desenvolvimento de medicamentos

> Se os modelos não generalizam de verdade, essas decisões podem estar erradas.

> **Recomendações:**
> 1. Sempre verificar vazamento de compostos entre splits
> 2. Avaliar desbalanceamento por quinase
> 3. Usar splits que garantam generalização verdadeira
> 4. Reportar métricas em cenários de compostos E quinases novos
> 5. Sempre comparar com baselines simples como o Lookup"

---

## SLIDE 21: Conclusão Final

**O que dizer:**
> "Para concluir:

> Os modelos KNN e MLP NÃO estão aprendendo padrões complexos de interação entre quinases e compostos.

> Eles estão MEMORIZANDO devido a quatro fatores:
> 1. Vazamento massivo de compostos
> 2. Desbalanceamento extremo por quinase
> 3. Comportamento consistente dos compostos
> 4. Alta similaridade química ao treino

> Esta análise é FUNDAMENTAL para validar qualquer modelo de predição de atividade quinase-composto. Antes de confiar em métricas impressionantes, precisamos garantir que o modelo está realmente generalizando."

---

## SLIDE 22: Obrigado / Perguntas

**O que dizer:**
> "Obrigado pela atenção. Estou à disposição para perguntas.

> Se quiserem explorar mais, todos os códigos e dados estão disponíveis, assim como os gráficos detalhados de cada análise."

---

## Perguntas Frequentes (FAQ)

### "Por que o MCC é mais importante que a acurácia?"
> A acurácia pode ser enganosa quando os dados são desbalanceados. Se 90% dos dados são de uma classe, um modelo que sempre responde essa classe tem 90% de acurácia mas não aprendeu nada. O MCC considera isso e só dá valores altos quando o modelo realmente distingue as classes.

### "O que é o KNN exatamente?"
> K-Nearest Neighbors (K-Vizinhos Mais Próximos) é um algoritmo simples: quando recebe um dado novo, ele procura os K dados mais parecidos no treino e usa a resposta da maioria. É como perguntar para seus vizinhos o que acharam de um restaurante - você confia na opinião de quem é mais parecido com você.

### "O que é um MLP?"
> Multi-Layer Perceptron é uma rede neural simples com algumas camadas de neurônios. É mais sofisticado que o KNN, mas ainda assim básico para os padrões atuais de deep learning.

### "Isso significa que machine learning não funciona para descoberta de medicamentos?"
> Não! Significa que precisamos ser MUITO cuidadosos com como preparamos os dados e avaliamos os modelos. Com curadoria adequada e splits corretos, podemos ter modelos que realmente generalizam - mas as métricas serão mais modestas e honestas.

### "Como deveríamos fazer o split correto?"
> O ideal é garantir que compostos e quinases no teste nunca apareçam no treino. Além disso, devemos garantir que compostos quimicamente similares também não apareçam em splits diferentes (cluster-based split). Isso simula o cenário real de uso.

---

## Referências Visuais

- **Slide 5**: Gráfico `01_leakage_analysis.png`
- **Slide 7**: Gráfico `02_baseline_comparison.png`
- **Slide 9**: Gráfico `03_kinase_imbalance.png`
- **Slide 11**: Gráfico `04_compound_consistency.png`
- **Slide 13**: Gráfico `05_similarity_analysis.png`
- **Slide 16**: Gráfico `06_split_comparison.png`
- **Slide 17**: Gráfico `07_inflated_vs_real_performance.png`

---

*Roteiro preparado para a apresentação "Por que Representações Simples Apresentam Performance Elevada?" - LNCC*
