add_subdirectory(core)
add_subdirectory(config)
add_subdirectory(image)
add_subdirectory(tensor)
add_subdirectory(preprocess)
add_subdirectory(postprocess)
add_subdirectory(engine)
add_subdirectory(pipeline)

add_library(ai_inference_lab INTERFACE)

target_link_libraries(ai_inference_lab
    INTERFACE
        ailab_core
        ailab_config
        ailab_image
        ailab_tensor
        ailab_preprocess
        ailab_postprocess
        ailab_engine
        ailab_pipeline
)