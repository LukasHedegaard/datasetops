Getting started
===============

.. code-block:: python
   import datasetops as do

   path = '../data/nested_class_folder'

   # Prepare your data
   train, val, test =                                       \
      do.load_folder_class_data(path)                      \
         .set_item_names('data','label')                    \  
         .as_img('data').resize((240,240)).as_numpy('data') \
         .one_hot('label')                                  \
         .shuffle(seed=42)                                  \
         .split([0.6,0.2,0.3])      

   # Do your magic using Tensorflow
   train_tf = trian.to_tf() 

   # Rule the world with PyTorch
   train_pt = trian.to_pytorch() #coming up!

   # Do your own thing
   for img, label in train:
      ...