<a name="readme-topo"></a>

<h1 align='center'>
  Atividade 03 - Redes Neurais e Sistemas Neuro-Fuzzy
</h1>

<div align='center'>

[![SO][Ubuntu-badge]][Ubuntu-url]
[![IDE][vscode-badge]][vscode-url]
[![Python][Python-badge]][Python-url]

<b>
  Guilherme Alvarenga de Azevedo<br>
  Maria Eduarda Teixeira Souza<br>
</b>
  
<br>
Inteligência Computacional <br>
Engenharia de Computação <br>
CEFET-MG Campus V <br>
2026/1


</div>

## Introdução

O presente projeto foi desenvolvido como parte da "Atividade 03 - Redes Neurais e Sistemas Neuro-Fuzzy" da disciplina de Inteligência Computacional. O foco principal deste trabalho é conduzir uma análise comparativa rigorosa entre arquiteturas puramente conexionistas (Redes Neurais Artificiais) [[2]](#ref2) e sistemas baseados em regras (Sistemas Neuro-Fuzzy) [[3]](#ref3). 

Foram avaliados quatro modelos: **MLP** (*Multilayer Perceptron*) [[4]](#ref4), **RBF** (*Radial Basis Function*) [[5]](#ref5), **ANFIS** (*Adaptive Neuro-Fuzzy Inference System*) [[6]](#ref6) e **FNN-FCM** (*Fuzzy Neural Network com Fuzzy C-Means*) [[7]](#ref7). O estudo avalia não apenas a precisão preditiva de cada modelo, mas também o custo computacional, a escalabilidade e o balanço entre modelos "caixa-preta" e a interpretabilidade das regras fuzzy.

## Base de Dados

**Tipo de Tarefa:** A atividade propõe uma tarefa de **Regressão Contínua**.

Para testar a robustez dos modelos frente a diferentes complexidades, volumetrias e dimensionalidades, os cenários de aplicação baseiam-se em 4 bases de dados reais distintas, obtidas na plataforma Kaggle:

1. **Superstore (Profit):** Previsão de lucro de vendas varejistas (9.994 amostras).
2. **Flight Price:** Previsão de preço de passagens aéreas, configurando a base massiva do estudo (300.153 amostras).
3. **Used Car Price:** Estimativa do preço de veículos usados, exigindo forte engenharia de atributos (4.009 amostras).
4. **Obesity (Weight):** Estimação de peso corporal a partir de hábitos e histórico familiar (2.111 amostras).

> [!NOTE] 
> Todas as bases passaram por um pipeline rigoroso de pré-processamento envolvendo limpeza, imputação de dados faltantes, codificação categórica (One-Hot e Label Encoding) e normalização via Min-Max Scaler.

## 📚 O Projeto

Neste repositório você encontrará o código-fonte do projeto, bem como os dados utilizados para a análise. O projeto foi desenvolvido integralmente em Python 3, com foco em modularidade e processamento paralelo.

A arquitetura do código foi projetada para garantir validade estatística e contornar gargalos de processamento. Os principais destaques da implementação incluem:

* **Otimização Automatizada:** Utilização da biblioteca `Optuna` para a busca sistemática dos melhores hiperparâmetros de cada modelo (15 *trials* guiados por validação cruzada 5-fold).
* **Validade Estatística:** O sistema executa 21 rodadas independentes para cada modelo, mitigando o viés da inicialização estocástica dos pesos.
* **Execução Distribuída:** Devido à explosão combinatória de regras do ANFIS em bases massivas, o pipeline (`src/main.py`) permite segmentar a carga de trabalho em perfis independentes.
* **Consolidação de Resultados:** O módulo `src/consolidate.py` varre as pastas de resultados geradas localmente em diferentes máquinas, calculando as médias e desvios-padrão das métricas (RMSE, MAE, R² e MAPE) e exportando o arquivo unificado `global_summary.csv`.

De uma forma compacta e organizada, os arquivos e diretórios estão dispostos da seguinte forma, onde cada item pode ser acessado clicando em seu respectivo link:

* 📁 **[`dataset/`](./dataset/)**
  * 📁 **[`data1/`](./dataset/data1/)**
  * 📁 **[`data2/`](./dataset/data2/)**
  * 📁 **[`data3/`](./dataset/data3/)**
  * 📁 **[`data4/`](./dataset/data4/)**
  * 📁 **[`processed/`](./dataset/processed/)**
* 📁 **[`documentacao/`](./documentacao/)**
  * 📄 [`artigo.pdf`](./documentacao/artigo.pdf)
  * 📄 [`slide.pdf`](./documentacao/slide.pdf)
* 📁 **[`results/`](./results/)**
  * 📁 **[`dataset_0/`](./results/dataset_0/)**
  * 📁 **[`dataset_1/`](./results/dataset_1/)**
  * 📁 **[`dataset_2/`](./results/dataset_2/)**
  * 📁 **[`dataset_3/`](./results/dataset_3/)**
  * 📁 **[`eda/`](./results/eda/)**
  * 📄 [`global_summary.csv`](./results/global_summary.csv)
* 📁 **[`src/`](./src/)**
  * 📁 **[`eda/`](./src/eda/)**
    * 📄 [`__init__.py`](./src/eda/__init__.py)
    * 📄 [`eda.py`](./src/eda/eda.py)
  * 📁 **[`models/`](./src/models/)**
    * 📄 [`__init__.py`](./src/models/__init__.py)
    * 📁 **[`neuro_fuzzy/`](./src/models/neuro_fuzzy/)**
      * 📄 [`__init__.py`](./src/models/neuro_fuzzy/__init__.py)
      * 📄 [`anfis.py`](./src/models/neuro_fuzzy/anfis.py)
      * 📄 [`fnn_fcm.py`](./src/models/neuro_fuzzy/fnn_fcm.py)
    * 📁 **[`rna/`](./src/models/rna/)**
      * 📄 [`__init__.py`](./src/models/rna/__init__.py)
      * 📄 [`mlp.py`](./src/models/rna/mlp.py)
      * 📄 [`rbf.py`](./src/models/rna/rbf.py)
  * 📁 **[`preprocessing/`](./src/preprocessing/)**
    * 📄 [`__init__.py`](./src/preprocessing/__init__.py)
    * 📄 [`preprocessing.py`](./src/preprocessing/preprocessing.py)
  * 📁 **[`utils/`](./src/utils/)**
    * 📄 [`__init__.py`](./src/utils/__init__.py)
    * 📄 [`metrics.py`](./src/utils/metrics.py)
    * 📄 [`plots.py`](./src/utils/plots.py)
  * 📄 [`config.py`](./src/config.py)
  * 📄 [`consolidate.py`](./src/consolidate.py)
  * 📄 [`experiments.py`](./src/experiments.py)
  * 📄 [`main.py`](./src/main.py)
* 📄 [`.gitignore`](./.gitignore)
* 📄 [`README.md`](./README.md)
* 📄 [`requirements.txt`](./requirements.txt)
* 📄 [`test_preprocessing.py`](./test_preprocessing.py)

## Instalando
Para instalar o projeto, siga os passos abaixo:

<div align="justify">
  Com o ambiente preparado, os seguintes passos são para a instalação, compilação e execução do programa localmente:

  1. Clone o repositório no diretório desejado:
  ```console
  git clone https://github.com/alvarengazv/rna-neuro-fuzzy-ic.git
  cd rna-neuro-fuzzy-ic
  ```
  2. Crie e ative um ambiente virtual (recomendado) - garanta que já possui o [Python](https://www.python.org/downloads/), no mínimo na versão 3.11.9:
  ```console
  python3 -m venv venv
  source venv/bin/activate   # Linux/macOS
  venv\Scripts\activate      # Windows
  ```
  3. Instale as dependências com pip: 
  ```console
    pip install -r requirements.txt
  ```
</div>
<div align="justify">
  
  4. Execute o programa:
      - **Linux/macOS**
        ```console
          # Usando Python diretamente
          # PYTHONPATH='src' python3 -m main
        ```

      - **Windows**
        ```console
          # Usando Python diretamente
          # python3 src/main.py
        ```
</div> 

<div align="justify">
  
  ## Dependências

  O projeto utiliza as seguintes bibliotecas:

  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - scikit-fuzzy
  - imblearn

</div>

> [!NOTE]
> No arquivo [`requirements.txt`](requirements.txt) tem todas essas informações.

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

## 🧪 Ambiente de Compilação e Execução

<div align="justify">

  O trabalho foi desenvolvido e testado em várias configurações de hardware. Podemos destacar algumas configurações de Sistema Operacional e Compilador, pois as demais configurações não influenciam diretamente no desempenho do programa.

</div>

<div align='center'>

[![SO][Ubuntu-badge]][Ubuntu-url]
[![IDE][vscode-badge]][vscode-url]
[![Python][Python-badge]][Python-url]

| *Hardware* | *Especificações* |
|:------------:|:-------------------:|
| *Laptop*   | Dell Inspiron 13 5330 |
| *Processador* | Intel Core i7-1360P |
| *Memória RAM* | 16 GB DDR5 |
| *Sistema Operacional* | Ubuntu 24.04 |
| *IDE* | Visual Studio Code |
| *Placa de Vídeo* | Intel Iris Xe Graphics |

</div>

> [!IMPORTANT] 
> Para que os testes tenham validade, considere as especificações
> do ambiente de compilação e execução do programa.

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

## 📨 Contato

<div align="center">
  <br><br>
     <i>Guilherme Alvarenga de Azevedo - Graduando - 7º Período de Engenharia de Computação @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor1]
  [![Linkedin][linkedin-badge]][linkedin-autor1]
  [![Telegram][telegram-badge]][telegram-autor1]
  
  
  <br><br>
     <i>Maria Eduarda Teixeira Souza - Graduando - 7º Período de Engenharia de Computação @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor2]
  [![Linkedin][linkedin-badge]][linkedin-autor2]
  [![Telegram][telegram-badge]][telegram-autor2]

</div>

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

<a name="referencias">📚 Referências</a>

1. <a name="ref1"></a>AZEVEDO, Guilherme A. SOUZA, Maria E. T. **RNA-NEURO-FUZZY-IC**: Atividade 03 - Redes Neurais e Sistemas Neuro-Fuzzy. 2026. Disponível em: [https://github.com/alvarengazv/rna-neuro-fuzzy-ic](https://github.com/alvarengazv/rna-neuro-fuzzy-ic) Acesso em: 15 jun. 2026.

2. <a name="ref2"></a>SILVA, Alisson M. **Inteligência Computacional**: Notas de Aula 11 - Redes Neurais. 2026.

3. <a name="ref3"></a>SILVA, Alisson M. **Inteligência Computacional**: Notas de Aula 12 - Neuro Fuzzy. 2026.

4. <a name="ref4"></a>RUMELHART, D. E.; HINTON, G. E.; WILLIAMS, R. J. Learning representations by back-propagating errors. *Nature*, vol. 323, no. 6088, pp. 533-536, 1986.

5. <a name="ref5"></a>MOODY, J.; DARKEN, C. J. Fast learning in networks of locally-tuned processing units. *Neural Computation*, vol. 1, no. 2, pp. 281-294, 1989.

6. <a name="ref6"></a>JANG, J.-S. R. ANFIS: adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 23, no. 3, pp. 665-685, 1993.

7. <a name="ref7"></a>LIN, C.-T.; LEE, C. S. G. Neural-network-based fuzzy logic control and decision system. *IEEE Transactions on Computers*, vol. 40, no. 12, pp. 1320-1336, 1991.


[vscode-badge]: https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white
[vscode-url]: https://code.visualstudio.com/docs/?dv=linux64_deb
[make-badge]: https://img.shields.io/badge/_-MAKEFILE-427819.svg?style=for-the-badge
[make-url]: https://www.gnu.org/software/make/manual/make.html
[cpp-badge]: https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white
[cpp-url]: https://en.cppreference.com/w/cpp
[trabalho-url]: https://drive.google.com/file/d/1-IHbGaA1BIC6_CMBydOC-NbV2bCERc8r/view?usp=sharing
[github-prof]: https://github.com/mpiress
[main-ref]: src/main.cpp
[branchAMM-url]: https://github.com/alvarengazv/trabalhosAEDS1/tree/AlgoritmosMinMax
[makefile]: ./makefile
[bash-url]: https://www.hostgator.com.br/blog/o-que-e-bash/
[lenovo-badge]: https://img.shields.io/badge/lenovo%20laptop-E2231A?style=for-the-badge&logo=lenovo&logoColor=white
[ubuntu-badge]: https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white
[Ubuntu-url]: https://ubuntu.com/
[ryzen5500-badge]: https://img.shields.io/badge/AMD%20Ryzen_5_5500U-ED1C24?style=for-the-badge&logo=amd&logoColor=white
[ryzen3500-badge]: https://img.shields.io/badge/AMD%20Ryzen_5_3500X-ED1C24?style=for-the-badge&logo=amd&logoColor=white
[windows-badge]: https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white
[gcc-badge]: https://img.shields.io/badge/GCC-5C6EB8?style=for-the-badge&logo=gnu&logoColor=white
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/


[linkedin-autor1]: https://www.linkedin.com/in/guilherme-alvarenga-de-azevedo-959474201/
[telegram-autor1]: https://t.me/alvarengazv
[gmail-autor1]: mailto:gui.alvarengas234@gmail.com

[linkedin-autor2]: https://www.linkedin.com/in/dudatsouza/
[telegram-autor2]: https://t.me/dudat_18
[gmail-autor2]: mailto:dudateixeirasouza@gmail.com

[linkedin-badge]: https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=Linkedin&logoColor=white
[telegram-badge]: https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white
[gmail-badge]: https://img.shields.io/badge/-Gmail-D14836?style=for-the-badge&logo=Gmail&logoColor=white
