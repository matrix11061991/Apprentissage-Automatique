
import tensorflow as tf

# Chargement du modèle pré-entraîné

model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Congélation des couches du modèle

for layer in model.layers:

    layer.trainable = False

# Ajout de nouvelles couches de classification

x = model.output

x = neural.layers.Flatten()(x)

x = neural.layers.Dense(1024, activation='relu')(x)

predictions = neural.layers.Dense(num_classes, activation='softmax')(x)

# Création du nouveau modèle

new_model = neural.Model(inputs=model.input, outputs=predictions)

# Compilation et entraînement du nouveau modèle

new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

new_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
