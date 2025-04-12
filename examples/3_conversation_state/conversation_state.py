"""
Example 3: Managing Conversation State and Context Across Interactions

This example demonstrates how an agent can maintain state across multiple interactions
and how context can be used to store and retrieve information during a conversation.
"""
import asyncio

from aikernel import Conversation, LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import BaseModel, Field

import sys
import os
# Add the parent directory to the path so we can import the custom_agent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_agent import Agent
from frizz import tool


# Models for the shopping cart tool
class Product(BaseModel):
    """Product information."""
    id: str
    name: str
    price: float
    category: str


class AddToCartParams(BaseModel):
    """Parameters for adding an item to the cart."""
    product_id: str
    quantity: int = Field(ge=1)


class RemoveFromCartParams(BaseModel):
    """Parameters for removing an item from the cart."""
    product_id: str
    quantity: int | None = Field(None, ge=1)


class ViewCartParams(BaseModel):
    """Parameters for viewing the cart (empty as no parameters needed)."""
    pass


class CartItem(BaseModel):
    """An item in the shopping cart."""
    product: Product
    quantity: int
    total_price: float


class CartResult(BaseModel):
    """Return type for all cart operations."""
    items: list[CartItem]
    item_count: int
    total_price: float
    message: str


# Context that persists across conversations
class ShoppingContext:
    """Shopping context that maintains the product catalog and cart state."""
    def __init__(self):
        # Product catalog
        self.products: dict[str, Product] = {
            "p1": Product(id="p1", name="Laptop", price=999.99, category="Electronics"),
            "p2": Product(id="p2", name="Smartphone", price=699.99, category="Electronics"),
            "p3": Product(id="p3", name="Headphones", price=149.99, category="Electronics"),
            "p4": Product(id="p4", name="Coffee Maker", price=79.99, category="Kitchen"),
            "p5": Product(id="p5", name="Blender", price=49.99, category="Kitchen"),
            "p6": Product(id="p6", name="T-shirt", price=19.99, category="Clothing"),
            "p7": Product(id="p7", name="Jeans", price=59.99, category="Clothing"),
        }
        
        # Shopping cart - persists between interactions
        self.cart: dict[str, CartItem] = {}
    
    def add_to_cart(self, product_id: str, quantity: int) -> CartResult:
        """Add a product to the cart."""
        if product_id not in self.products:
            raise ValueError(f"Product with ID {product_id} not found")
        
        product = self.products[product_id]
        
        if product_id in self.cart:
            # Update existing cart item
            self.cart[product_id].quantity += quantity
            self.cart[product_id].total_price = self.cart[product_id].quantity * product.price
            message = f"Added {quantity} more {product.name} to your cart"
        else:
            # Create new cart item
            self.cart[product_id] = CartItem(
                product=product,
                quantity=quantity,
                total_price=quantity * product.price
            )
            message = f"Added {quantity} {product.name} to your cart"
        
        return self._get_cart_state(message)
    
    def remove_from_cart(self, product_id: str, quantity: int | None = None) -> CartResult:
        """Remove a product from the cart."""
        if product_id not in self.cart:
            raise ValueError(f"Product with ID {product_id} not in your cart")
        
        product_name = self.cart[product_id].product.name
        
        if quantity is None or quantity >= self.cart[product_id].quantity:
            # Remove entire item
            del self.cart[product_id]
            message = f"Removed all {product_name} from your cart"
        else:
            # Update quantity
            self.cart[product_id].quantity -= quantity
            self.cart[product_id].total_price = (
                self.cart[product_id].quantity * self.cart[product_id].product.price
            )
            message = f"Removed {quantity} {product_name} from your cart"
        
        return self._get_cart_state(message)
    
    def view_cart(self) -> CartResult:
        """View the current cart contents."""
        if not self.cart:
            message = "Your cart is empty"
        else:
            message = "Here's your current shopping cart"
        
        return self._get_cart_state(message)
    
    def _get_cart_state(self, message: str) -> CartResult:
        """Helper to generate the cart state."""
        items = list(self.cart.values())
        item_count = sum(item.quantity for item in items)
        total_price = sum(item.total_price for item in items)
        
        return CartResult(
            items=items,
            item_count=item_count,
            total_price=total_price,
            message=message
        )


# Define the tool for adding to cart
@tool(name="add_to_cart")
async def add_to_cart(
    *, context: ShoppingContext, parameters: AddToCartParams, conversation: Conversation
) -> CartResult:
    """Add a product to the shopping cart."""
    return context.add_to_cart(parameters.product_id, parameters.quantity)


# Define the tool for removing from cart
@tool(name="remove_from_cart")
async def remove_from_cart(
    *, context: ShoppingContext, parameters: RemoveFromCartParams, conversation: Conversation
) -> CartResult:
    """Remove a product from the shopping cart."""
    return context.remove_from_cart(parameters.product_id, parameters.quantity)


# Define the tool for viewing the cart
@tool(name="view_cart")
async def view_cart(
    *, context: ShoppingContext, parameters: ViewCartParams, conversation: Conversation
) -> CartResult:
    """View the current shopping cart contents."""
    return context.view_cart()


async def main():
    # Create a shared context that will persist across interactions
    shopping_context = ShoppingContext()
    
    # Create an agent with the cart tools
    agent = Agent(
        tools=[add_to_cart, remove_from_cart, view_cart],
        context=shopping_context,
        system_message=LLMSystemMessage(parts=[LLMMessagePart(content="""
            You are a shopping assistant that helps users manage their shopping cart.
            When users ask to add or remove items, use the appropriate tools.
            When users ask about what's in their cart, use the view_cart tool.
            Always refer to products by their exact name as shown in the catalog.
        """)])
    )
    
    # Create a router for the LLM API
    router = get_router(models=("gemini-2.0-flash",))
    
    # Example conversation that demonstrates state persistence
    print("Starting conversation with the shopping assistant...\n")
    
    # First interaction: Add to cart
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="I'd like to add a Laptop and two Headphones to my cart")])
    print(f"User: {user_message.parts[0].content}")
    
    result = await agent.step(
        user_message=user_message,
        model="gemini-2.0-flash",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Second interaction: Check cart contents
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="What's in my cart now?")])
    print(f"\nUser: {user_message.parts[0].content}")
    
    result = await agent.step(
        user_message=user_message,
        model="gemini-2.0-flash",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Third interaction: Remove an item
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="Remove one of the headphones please")])
    print(f"\nUser: {user_message.parts[0].content}")
    
    result = await agent.step(
        user_message=user_message,
        model="gemini-2.0-flash",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Fourth interaction: Add more items
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="Add a Coffee Maker and a T-shirt")])
    print(f"\nUser: {user_message.parts[0].content}")
    
    result = await agent.step(
        user_message=user_message,
        model="gemini-2.0-flash",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Final interaction: Check the updated cart
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="Show me my final cart")])
    print(f"\nUser: {user_message.parts[0].content}")
    
    result = await agent.step(
        user_message=user_message,
        model="gemini-2.0-flash",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")


if __name__ == "__main__":
    asyncio.run(main())
