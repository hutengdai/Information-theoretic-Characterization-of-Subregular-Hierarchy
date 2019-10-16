# Information-theoretic-Characterization-of-Subregular-Hierarchy
This is the repository for the project of Information-theoretic Characterization of Subregular Hierarchy

Prerequisites: Python 3, PyTorch, IPython. 



Usage:
- Step 1: Install all modules in rfutils from https://github.com/Futrell/rfutils. 

- Step 2: Github clone the repository to your directory 
- Step 3: Open IPython
```bash
ipython
```
- Step 4: Change directory to the folder which contain pfa.py
```bash
cd [your directory] 
```

- Step 5: In IPython, import pfa.py as a module
```bash
import pfa
```
- Step 6: Choose which Probabilistic Finite-state Acceptor (PDFA) you are working on, for example, a strictly 2-local PDFA corresponds to the built-in function no_ab_example(). 
```bash
sl2 = pfa.no_ab_example()
```
- Step 7: Compute Statistical Complexity
```bash
sl2.statistical_complexity
```
- Step 8: Compute the bound of Crypticity using sl2.crypticity_estimate(step). 
```bash
sl2.crypticity_estimate(3)
sl2.crypticity_estimate(4)
......
sl2.crypticity_estimate(7)
sl2.crypticity_estimate(8)
......
#Iterate this process until reach the limit of your computer power when your IPython drops
```

- Step 9: Compute the bound of Excess Entropy using sl2.excess_entropy_estimate(step). 
```bash
sl2.excess_entropy_estimate(3)
sl2.excess_entropy_estimate(4)
......
sl2.excess_entropy_estimate(7)
sl2.excess_entropy_estimate(8)
......
#Iterate this process until reach the limit of your computer power when your IPython drops
```

The built-in PDFAs (functions);
1. Minimal SL2: no_ab_example()
2. Canonical SL2: no_ab_canonical_example()
3. Minimal LT2: exists_ab_example()
4. Canonical LT2: exists_ab_canonical_example()
5. Minimal LTT2: one_ab_example()
6. Canonical LTT2: one_ab_canonical_example()
7. Minimal SP2: strictly_piecewise_example()
8. Minimal PT2: piecewise_testable_example()
9. Minimal SL3: strictly_local_three_example()
10. *NT (Pater 2016) SL2: sl2_nt_example()

To test your own Probabilistic Finite-state Acceptors, you should write a function in Python


- Feel free to contact me (hutengdai[*]gmail.com) if you have any questions!
