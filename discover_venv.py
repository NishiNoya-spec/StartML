import sys

if __name__ == "__main__":

    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Вы находитесь в виртуальном окружении
        if hasattr(sys, 'base_prefix'):
            print(f"Вы находитесь в виртуальном окружении: {sys.base_prefix}")
        else:
            print("Вы находитесь в виртуальном окружении (старый стиль)")
    else:
        # Вы находитесь в системном окружении
        print("Вы находитесь в системном окружении")