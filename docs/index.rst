tether
======

High-precision record linkage library implementing a 7-step pipeline with multi-pass support.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Installation
------------

.. code-block:: bash

   pip install tether

Quick Start
-----------

.. code-block:: python

   import pandas as pd
   from tether import Pipeline, StringComparison, ExactComparison

   df_left = pd.DataFrame({
       "id": [1, 2, 3],
       "first_name": ["John", "Jane", "Bob"],
       "last_name": ["Smith", "Doe", "Johnson"],
       "state": ["CA", "NY", "CA"],
   })

   df_right = pd.DataFrame({
       "id": [101, 102, 103],
       "first_name": ["Jon", "Jane", "Robert"],
       "last_name": ["Smith", "Doe", "Johnson"],
       "state": ["CA", "NY", "CA"],
   })

   result = (
       Pipeline()
       .preprocess(normalize_unicode=True, lowercase=True)
       .block(on="state")
       .score(comparisons=[
           StringComparison("first_name", algorithm="jaro_winkler"),
           StringComparison("last_name", algorithm="jaro_winkler"),
       ])
       .filter(min_score=0.7)
       .decide(method="hungarian")
       .build()
       .link(df_left, df_right)
   )

   print(result.matches)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
