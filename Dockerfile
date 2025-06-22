# Use the official Blender 4.4 headless image
FROM linuxserver/blender:arm64v8-4.4.3

# Put everything under /app
WORKDIR /app

# Copy just the requirements so Docker can cache this layer
COPY requirements.txt .

# Bootstrap pip into Blender’s Python and install deps
RUN blender --background --python-expr "\
    import ensurepip; ensurepip.bootstrap(); \
    import pip; pip.main(['install', '-r', 'requirements.txt'])"

# Copy the rest of the project
COPY . .
