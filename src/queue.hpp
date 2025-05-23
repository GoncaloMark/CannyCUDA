#pragma once 
#include <stdexcept>

/**
    * @class Queue
    * @brief A generic queue implementation using a linked list structure.
    * 
    * This class provides a simple queue data structure with FIFO (First In, First Out) behavior. It supports operations to add elements to the end of the queue and remove elements from the front of the queue.
    * 
    * @tparam T The type of elements stored in the queue
*/
template <typename T>
class Queue{
    /**
        * @class Node
        * @brief Internal node structure for the linked list implementation of the queue.
        * 
        * Each node contains an element and a pointer to the next node in the list.
    */
    class Node{
        public:
            /**
                * @brief Constructs a new Node with the given element.
                * 
                * @param element The element to store in the node
            */
            explicit Node(T&& element) : element(std::forward<T>(element)), next(nullptr){};
            
            /**
                * @brief The element stored in the node.
            */
            T element;
            
            /**
                * @brief Pointer to the next node in the linked list.
            */
            Node* next;
    };

    private:
        /**
            * @brief Pointer to the first node in the queue.
        */
        Node* first;
        
        /**
            * @brief Pointer to the last node in the queue.
        */
        Node* last;

    public:
        /**
            * @brief Constructs an empty queue.
        */
        explicit Queue() : first(nullptr), last(nullptr){};

        /**
            * @brief Checks if the queue is empty.
            * 
            * @return true if the queue is empty, false otherwise
        */
        inline bool isEmpty() const {
            return first == nullptr;
        };

        /**
            * @brief Adds an element to the end of the queue.
            * 
            * @param element The element to add to the queue
        */
        void enQueue(T&& element);
        
        /**
            * @brief Removes and returns the element at the front of the queue.
            * 
            * @return The element at the front of the queue
            * @throws std::runtime_error if the queue is empty
        */
        T deQueue();
        
        /**
            * @brief Removes all elements from the queue.
            * 
            * Repeatedly calls deQueue() until the queue is empty.
            * 
        */
        inline void clear(){
            while(!isEmpty())
                deQueue();
        }
};

/**
    * @brief Adds an element to the end of the queue.
    * 
    * Creates a new node with the provided element and adds it to the end of the queue.
    * If the queue is empty, both first and last pointers are updated to point to the new node.
    * 
    * @tparam T The type of elements stored in the queue
    * @param element The element to add to the queue
*/
template <typename T>
void Queue<T>::enQueue(T&& element) {  
    if (isEmpty()) {
        first = new Node(std::forward<T>(element));  
        last = first;
    } else {
        Node* p = new Node(std::forward<T>(element));  
        last->next = p;
        last = last->next;
    }
}

/**
    * @brief Removes and returns the element at the front of the queue.
    * 
    * Removes the first node from the queue and returns its element.
    * Updates the first pointer to point to the next node.
    * If the queue becomes empty, updates the last pointer to nullptr.
    * 
    * @tparam T The type of elements stored in the queue
    * @return The element at the front of the queue
    * @throws std::runtime_error if the queue is empty
*/
template <typename T>
T Queue<T>::deQueue() {
    if (isEmpty()) {
        throw std::runtime_error("Queue is empty!"); 
    }

    Node *p = first;
    T element = std::move(p->element);  

    first = first->next;
    delete p;

    if (first == nullptr) {
        last = nullptr;  
    }

    return std::move(element);
}