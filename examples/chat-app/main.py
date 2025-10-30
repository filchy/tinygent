import tiny_chat as tc


@tc.on_message
def handle_message(msg: str):
    return f'You said: {msg}'


if __name__ == '__main__':
    tc.run(reload=True)
