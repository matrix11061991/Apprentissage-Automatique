
import tensorflow as tf

# Chargement du modèle pré-entraîné

base_model = tf.keras.applications.ResNet50(weights='imagenet',

                                               include_top=False,

                                               input_shape=(150, 150, 3))

# Ajout de nouvelles couches de classification au-dessus du modèle de base

x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(1024, activation='relu')(x)

predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Création du modèle final en combinant le modèle de base et les nouvelles couches

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Congélation des poids du modèle de base (on ne les mettra pas à jour pendant l'entraînement)

for layer in base_model.layers:

    layer.trainable = False

# Compilation et entraînement du modèle final

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(X_val, y_val))
