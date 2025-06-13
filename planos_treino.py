class PlanoTreino:
    def __init__(self, nome, exercicios):
        self.nome = nome
        self.exercicios = exercicios
        self.exercicio_atual_index = 0

    def exercicio_atual(self):
        """Retorna o exercício atual do plano"""
        if not self.exercicios:
            return None
        return self.exercicios[self.exercicio_atual_index]

    def proximo_exercicio(self):
        """Avança para o próximo exercício do plano. Retorna o próximo exercício ou None se terminou."""
        if self.exercicio_atual_index < len(self.exercicios) - 1:
            self.exercicio_atual_index += 1
            return self.exercicios[self.exercicio_atual_index]
        return None  # Já chegou ao fim

    def reiniciar(self):
        """Reinicia o plano do primeiro exercício"""
        self.exercicio_atual_index = 0
        return self.exercicio_atual()

# Definição dos planos de treino disponíveis
PLANOS_DISPONIVEIS = {
    "Treino Completo": [
        ("Agachamento", 45),
        ("Flexao", 45),
        ("Prancha", 60),
        ("Lunge", 45),
        ("Jumping Jacks", 60),
        ("Abdominais", 45)
    ],
    "Treino Superior": [
        ("Flexao", 60),
        ("Prancha", 60),
        ("Abdominais", 45)
    ],
    "Treino Inferior": [
        ("Agachamento", 45),
        ("Lunge", 45),
        ("Jumping Jacks", 60)
    ],
    "Treino Cardio": [
        ("Jumping Jacks", 60),
        ("Agachamento", 45),
        ("Lunge", 45)
    ],
    "Treino Força": [
        ("Flexao", 45),
        ("Agachamento", 45),
        ("Abdominais", 45)
    ]
} 