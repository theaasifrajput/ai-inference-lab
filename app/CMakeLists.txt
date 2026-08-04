add_executable(ai_lab
    main.cpp
)

target_link_libraries(ai_lab
    PRIVATE
        ai_inference_lab
)